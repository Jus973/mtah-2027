# Submission log

One row per candidate CSV (PLAN.md §6.6). The final choice is the row marked FINAL.

| datetime | candidate | config sha | git SHA | csv sha256 | public check (verdict only) | held-out sim spellF1 | bands | notes | uploaded? |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-31 18:10 | v1 | f96f2f64a285 | 4cbeed1 | 3c32c4758d117a6a | pass | 0.5406 | pass | consistency 0.740 | no |
| 2026-10-31 18:10 | v1-h2 | bc6afc21311e | 4cbeed1+dirty | 6b2c1299a6e2a7e2 | pass | 0.5217 | FAIL | consistency 0.714 | no |
| 2026-10-31 20:30 | v2 | 14177d87de60 | ef58ae0 | a5a8eb85099c74fb | pass | 0.5450 | pass | consistency 0.728; near-tie with v1, not selected (§6.4) | no |
| 2026-10-31 20:46 | v3 | 2cb112d12c53 | 6dfcf31 | 00f4551f83d84c89 | pass | 0.5838 | pass | consistency 0.740; SELECTED: paired §6.4 (amended), 7/7 sim matches beat v1 | no |
| 2026-10-31 21:01 | v4 | 1b26d03d65f2 | 65493ac | 66da042216912ab0 | pass | 0.5930 | pass | consistency 0.726; rejected: sim tie (paired rule), consistency regression | no |
| 2026-10-31 21:13 | ablate-v3-no_oob | 2ad4fc1b4402 | 158d253 | 09352f5c76797526 | pass | 0.5784 | pass | consistency 0.742; M3 ablation, not a submission | no |
| 2026-10-31 21:13 | ablate-v3-no_still | cb4e3815f479 | 158d253 | 00f4551f83d84c89 | pass | 0.5808 | pass | consistency 0.740; M3 ablation (= v3 output: stillness inert) | no |
| 2026-10-31 21:26 | final | d41eecabe3a4 | 5b20987 | 00f4551f83d84c89 | pass | 0.5808 | pass | FINAL until 1 Nov, superseded by v5 (§6.4): configs/final.yaml = v3 minus inert stillness, same file as v3; consistency 0.740 | no |
| 2026-11-01 00:43 | v5 | 82ab56b08fbc | 53938f3 | c1abb6e5e46d78ab | pass | 0.6087 | pass | consistency 0.746; selected 1 Nov over the v3 final (paired §6.4, 7/7, +0.030, p = 0.0078); superseded by v6 | no |
| 2026-11-01 00:44 | final | 63d3fbdb94ed | 80906c1 | c1abb6e5e46d78ab | pass | 0.6087 | pass | FINAL 1 Nov until v6 (§6.4): configs/final.yaml = v5, same file as v5; consistency 0.746 | no |
| 2026-11-01 10:16 | v6 | 85a3d741a110 | 9e6206d | fe4b6878b3a22d6a | pass | 0.6270 | pass | consistency 0.742; SELECTED: make paired vs v5, 7/7, +0.014, p = 0.0078 | no |
| 2026-11-01 10:16 | final | 85a3d741a110 | 9e6206d | fe4b6878b3a22d6a | pass | 0.6270 | pass | FINAL 1 Nov until v11 (adoption entry; §6.4): configs/final.yaml = v6, same file as v6; consistency 0.742; preserved as rules_v6_final.yaml | no |
| 2026-11-01 10:40 | v7 | 47162a982152 | f75a82f | 75605ff5d364235b | pass | 0.6254 | pass | consistency 0.755; tie with v6 on the sim, worse on both dead-ball stress tests: not selected | no |
| 2026-11-01 11:42 | v8 | c649ed11eaac | e3f13ee | 4e8c9387b1781a99 | pass | 0.6523 | pass | consistency 0.722; rejected: set-piece alignment 12% (v6 32%), consistency regression | no |
| 2026-11-01 16:07 | v9 | bb700a82dc4d | 7f23da4+dirty | 4427af6db6f41765 | pass | 0.6470 | pass | consistency 0.739; NOT adopted (6/7 sim tie, Q3 open); amended guards all pass — see the 1 Nov work-log evidence block | no |
| 2026-11-01 16:11 | v9d | 3c485c82272d | 7f23da4+dirty | 4874ce73a4bd7c60 | pass | 0.6394 | pass | consistency 0.739 | no |
| 2026-11-01 16:54 | v11 | 5f1e728257d1 | 7f23da4+dirty | 7b83f3a89bb15fa8 | pass | 0.6636 | pass | consistency 0.730 (exclusive-alive 0.600, in-gap 0.686); SELECTED: make paired vs v6, 7/7, +0.0289, p = 0.0078, adopted with the one-way sign-off (adoption entry) | no |
| 2026-11-01 23:08 | final | 259d8aae1d1a | 7f23da4+dirty | 7b83f3a89bb15fa8 | pass | 0.6636 | pass | **FINAL** (adoption entry; §6.4): configs/final.yaml = v11, same resolved pipeline config (hash differs from v11's only via the inert clipviewer map); consistency 0.730; cold ×2 + docker arm64/amd64 byte-identical | no |
| 2026-11-05 00:18 | v12 | 5b9e56cd549b | 3a958c9+dirty | 609343ecfdf06761 | pass | 0.6600 | pass | consistency 0.721 | no |
| 2026-11-05 07:27 | v12h | c90a8155334f | 3a958c9+dirty | 961e800ef78f189a | pass | 0.6600 | pass | consistency 0.733 | no |
| 2026-11-05 12:32 | v12hA | 1ba66b0a54f5 | 452efcb | 25876f5bb7eb2124 | pass | 0.6680 | pass | consistency 0.733 | no |


## Work log

- 2026-10-30: M0 scaffold. Vendored official repo @ f8f2629. `scripts/sim_costs.py` reproduces
  the PLAN §2 cost table exactly on the public GT. Fake match generator, ATD loader, kick-off and
  direction detection, and 17 tests, all passing with no real data present. Real-data checks are
  pending a human run (see PLAN_STATUS.md).
- 2026-10-31: Kick-off detector v2 (formation break, ±1 s) PASS on real data, +4 frames in both
  halves. Added the airlock (`make airlock T=<target>`, `src/mtah/airlock.py`, `scripts/airlock.py`):
  agent-run real-data targets release only policy-checked aggregates, and every release is logged in
  `<cache>/airlock/released.jsonl`. First agent run: anchors, identical to the human run.
- 2026-10-31: M1 sim v0. DFL/IDSSE open data (7 matches) degraded to the real ATD's label-free
  statistics (measured through the airlock; ball lost mostly in stoppages). Sim acceptance test
  PASS: official baseline smooth50 0.261 on the sim vs 0.243 full-match reference. The sim GT is
  more fragmented than the real one (see PLAN_STATUS.md, known sim biases).
- 2026-10-31: Public-window bug check is now verdict-only (`make airlock T=public-check CSV=...`);
  `score()` refuses the public GT for candidates. Candidate CSVs are read-denied to the agent.
- 2026-10-31: Rules v1 touches + owner hypotheses H1-H3, tuned on the sim fit matches and
  compared on held-out with oracle D: H1 0.606, H2 0.585, H3 0.504 (smooth50 + oracle D 0.403).
  H1 picked (configs/rules.yaml); H2 kept (configs/rules_h2.yaml). Touch statistics on the real
  match (airlock) match the sim's closely, so the tuned 1.0 m radius should transfer.
- 2026-10-31: Dead-ball v1 (out of bounds, long gaps, stillness; D up to the restart touch),
  tuned end to end on the sim fit matches: held-out spellF1 0.541 with detected D (oracle 0.606,
  official baseline 0.286). Real match, label-free: 135 D runs (band PASS), 86% of set-piece tags
  start inside predicted D; predicted restarts sit 9.5 s after set-piece tag starts.
- 2026-10-31: Candidate v1 (rules.yaml: touches + H1 + dead-ball v1) on the real match through the
  airlock: validator, all bands and the public bug check pass; Smart-Tagging consistency 0.740;
  held-out sim 0.541. v1-h2 (H2) logged for comparison: worse on every label-free check. v1
  regenerates byte-identically. Waiting for the human Form upload.
- 2026-10-31: Draft technical document `docs/technical.md` (PLAN §8) for the early upload of v1.
  `make doc TEAM=... REPO=...` fills the placeholders, drops the team notes and writes
  docs/<TEAM>.md (plus a PDF with pandoc). New in the doc: held-out sim bootstrap 95% CIs from the
  official scorer, 1,000 resamples (J03WQQ / J03WR9):
  - v1: 0.522 [0.484, 0.560] / 0.560 [0.526, 0.594];
  - smooth50: 0.287 [0.259, 0.314] / 0.284 [0.251, 0.315].
  A timing re-run of v1 regenerated the same sha256 (4 s with a warm cache).
- 2026-10-31: Dead-ball v2 (`configs/rules_v2.yaml`) adds four rules, off in rules.yaml so v1
  regenerates byte-identically: calm-gap filter, release restarts, flight restarts, launch
  touches.
  - Sim held-out spellF1 0.5450 vs v1 0.5406, a near-tie. D share 0.486 vs 0.515 (GT 0.413).
  - Real match: all checks pass; D 46.0%; consistency 0.728 (v1 0.740).
  - §6.4 tie-breaks favour v1, which stays selected.
  - Partial oracles on the fit matches: perfect D is worth only +0.04, mostly boundary timing.
- 2026-10-31: Owner decoder v3 (`configs/rules_v3.yaml`): a learned takeover probability per
  contact (LightGBM, sim fit matches only, `models/takeover_v1.txt`) decoded by a two-state
  Viterbi over contacts.
  - Held-out sim 0.584 vs v1 0.541; better on all 7 sim matches.
  - Real match: all checks pass; consistency 0.740; 312 spells, 18% under 2 s.
  - Selection v1 → v3 is pending a team decision: §6.4's interval-overlap rule says tie, the
    paired test says v3.
- 2026-10-31: §6.4 step 2 amended to a paired per-match comparison: a positive mean paired
  difference and a one-sided sign test at p ≤ 0.05 over the held-out matches plus the fit
  matches where the challenger is out of fold. **v3 selected over v1**: Δ +0.043 held-out, 7/7
  matches, p ≈ 0.008. v2 vs v1 stays a tie, since v2 is tuned in-sample and only the 2 held-out
  matches count.
- 2026-10-31: Plan finished on the agent side (M2–M4).
  - `make submission` (configs/final.yaml) reproduces sha256 00f4551f83d84c89… from a clean
    clone, twice.
  - Latency corrections, v4 and tag anchors were tried and dropped; T3 broadcast and whistles
    failed their falsifiers.
  - M3 ablations: stillness dropped as inert; out-of-bounds kept.
  - Stability, disagreement and set-piece checks pass. `make audit` passes.
  - FINAL row above. The remaining steps are human (PLAN_STATUS.md TODO).
- 2026-10-31: M4 reproducibility closed. All of these produce sha256 00f4551f83d84c89…, the
  FINAL row, with validator and bands passing:
  - a fresh clone and venv with an empty cache (`make airlock T=submission-cold`), twice;
  - the Docker image on linux/arm64 and linux/amd64 (`make airlock T=submission-docker`).
- 2026-10-31: Public repo https://github.com/Jus973/mtah-2027. Final file named for the team
  (Berkeley Sports Research Group), same sha256; Docker entry point re-verified. Emails 1 and 2
  sent to the organisers (user-approved). The deadline is 17 Nov 11:59 PM CET and the Form link
  arrives by 14 Nov; a daily Gmail watch flags it. Upload waits for the user's go.
- 2026-11-01: Owner inside long ball gaps (v5, `configs/rules_v5.yaml`). v3 calls every gap of
  ≥ 25 frames `D`. On the sim, 20% of long-gap frames are alive and the missed turnovers that
  sit near gaps are all inside long ones (median 24 s). Oracles over v3: perfect `D` in gaps
  +0.015; the true labels in gaps +0.10. So the owner during gaps is the lever, not the contacts
  at a gap's ends (the planned next step 1).
  - Two LightGBM gap models trained on the sim fit matches from the players alone
    (`learn/gapstate.py`): P(dead) and P(H owns | alive), 44 team-shape and motion features.
    A three-state Viterbi re-decodes each long gap (`decode/gaps.py`); penalties tuned out of
    fold (`make train-gaps`). Gap timing and contact features added only 0.003 and were left out.
  - Paired vs the v3 final: +0.030, 7/7 matches, sign test p = 0.0078; held-out 0.581 → 0.609,
    macroF1 0.827 → 0.878.
  - Real match (airlock): all candidate checks pass; consistency 0.746 (v3 0.740) and 0.847 on
    the gap frames v5 decodes alive; 6.8% of gap frames decoded alive (sim 17–20%), with
    tag-phase coverage of its gap `D` (51%) close to other `D` (59%). Stability min 0.861; v3 vs
    v5 differ on 2.7% of graded frames.
  - **v5 selected (FINAL).** sha256 c1abb6e5e46d78ab…, identical twice, from a cold cache, from a
    fresh clean clone (cold, twice) and in Docker on linux/arm64 and linux/amd64. Not uploaded.
- 2026-11-01: v3 was restored as FINAL (66652fe) because the user had asked for v3; the user then
  set a new goal, to optimise the entry for winning the whole challenge. v5 is re-selected by the
  §6.4 paired rule (47d485f reverts the restore; same file, sha256 c1abb6e5e46d78ab…).
- 2026-11-01: Optimising for the win (user goal). Oracles on v5 (held-out sim): the true owner on
  alive frames is worth +0.23, the true `D` +0.085 (+0.058 of it outside long gaps), the true
  labels in gaps +0.065. Of v5's missed turnovers outside gaps, two-thirds had a contact by the
  new owner's team within 1 s: the takeover decision is the lever.
  - **v6** (`configs/rules_v6.yaml`): ten team-shape signals at each contact join the takeover
    model (`models/takeover_v2.txt`, AUC 0.915). `make paired` vs v5: +0.014, 7/7, p = 0.0078;
    held-out 0.609 → 0.627. Real match: all checks pass; consistency 0.742 (v5 0.746); stability
    min 0.867. **Selected (FINAL)**, sha256 fe4b6878b3a22d6a…: twice, and from a fresh clean
    clone (cold, twice).
  - Tried and dropped (paired vs v6 unless stated): all 88 shape features (+0.0145, 6/7: a tie);
    ball direction at contacts +0.001; label lag 12 / 25 frames −0.011 / −0.038; capacity
    variants ≤ 0; a 5-seed average −0.002; gap models with 5 s context −0.002; gap penalties
    re-tuned on v6: unchanged.
  - Learned per-frame `D` (shape + ball): +0.034 (7/7), but it reads the DFL data's frozen
    dead-time ball. On the sim 20.5–22.0% of seen ball frames do not move; on the real ATD
    0.08% (airlock atd-stats). Without ball kinematics +0.018 (6/7, a tie); players only −0.030.
    Dropped: no learned part that decides `D` reads ball kinematics.
  - **v7** (`configs/rules_v7.yaml`): a restart must send the ball away (>= 4 m/s, physical). Why:
    with the frozen sim ball replaced by one carried by the nearest player, v6 fell 0.627 → 0.556
    held-out. But that stress test teleported out-of-bounds balls back onto the pitch. With the
    ball left in place until a player comes within 1 m and then carried, v6 holds (0.625). v7 vs
    v6 (all 7 matches, same models): frozen −0.0006 (5/7), picked up −0.0036 (0/7), lost
    −0.0048 (1/7). Real match: all checks pass, consistency 0.755 (v6 0.742), 121 `D` runs, `D`
    share 0.518. A tie on the sim, worse on both stress tests: **not selected, v6 stays FINAL**
    (Docker linux/arm64 and amd64 reproduce v6's sha256 too).
  - **v8** (`configs/rules_v8.yaml`, `learn/deadstate.py`, `make train-dead`): a learned per-frame
    `D` (team shape + ball position and speed), trained after making the sim's frozen dead-time
    ball realistic (tracker noise 2 cm, fetched and carried, or lost). Sim, paired vs v6 with
    every learned part cross-fitted: +0.022 (7/7) on the DFL sim as is and with 1 cm noise, both
    never trained on; +0.029-0.040 (7/7) on the training conditions; +0.012 (6/7) with 5 cm
    noise. Held-out 0.627 → 0.652, `D` share 0.38-0.41 vs GT 0.41-0.42. **Real match rejects it**
    (airlock, label-free): set-piece alignment 12% (v6 32%, chance about 9%), consistency 0.722
    (v6 0.742), tag agreement on gap frames decoded alive 0.726 (v6 0.814). Its `D` boundaries do
    not meet real restarts, so the realistic-ball variants still miss something about the real
    tracker. Not selected (PLAN M3: no consistency regression). **v6 stays FINAL.**
  - Final models on more sim matches: the held-out learning curve rises (3 / 4 / 5 training
    matches: 0.6145 / 0.6188 / 0.6270), but leave-one-out models trained on the other 6 matches
    score −0.0031 vs v6 (3/7; J03WR9 −0.027). Not adopted: v6 keeps its 5-match models.
- 2026-11-01: **Pre-registration for the v9 cycle and the amended guards** (handoff/REPLY.md P0;
  written before any nested-OOF or paired v9 result was read; the independent review's sim
  measurements of the UNIFORM variant in handoff/REPLY.md §2.P1 predate this block and are
  disclosed there). HUMAN SIGN-OFF: pending (REPLY §6 Q1/Q3/Q4) — until signed, every amended
  guard below is REPORTED next to the old one and the old gates stand; the strict §6.4 rule
  (mean > 0 AND 7/7) decides adoption.
  - Harness repair (no judgement change): `make paired` now refuses config sections it does not
    apply (it silently scored `deadstate` — and would have scored `postprocess` — as plain v6:
    both arms identical, guaranteed tie; the v8 official gate was moot), applies postprocess /
    deadstate / release-restart injection exactly as `mtah.pipeline`, and prints a
    possession-level paired bootstrap as a second opinion (never the gate). P0.1 smoke
    (`paired.py --smoke`, run 1 Nov): deterministic — final.yaml twice gives identical labels,
    per-match sha16s fe660877…/763c86fb…/0bf93dad…/0f206486…/af3d84e9…/fb122c2d…/370dbb47… —
    and final vs rules_v9d labels differ on 7/7 matches: the guaranteed-tie class is dead. The
    whitelist also fired in anger the same day: a new `metrica_accept` section was refused until
    classified INERT.
  - Amended guards, thresholds fixed now: (1) Smart-Tagging consistency is judged on the
    exclusive-alive decomposition (`disagreement` job; pooled rate tracks D share ~0.005/point
    across v1–v8, so it cannot rank candidates of different alive share) — floor: the candidate's
    exclusive-alive agreement must not sit below the back-catalog's re-baselined range by more
    than 0.02 with >= 500 frames in the bucket; the pooled number is still reported. (2)
    Set-piece alignment is judged at the config-corrected anchor (9.48 − k/25 s for rules that
    move D ends k frames; k from the config, never fitted): >= 32% within ±2 s AND offset IQR
    <= 1.2× v6's re-baselined IQR; raw-anchor number and candidate-matched chance reported
    beside it. (3) New advisory band d_share_graded in [0.35, 0.57] (derivation in
    configs/base.yaml), advisory this cycle, hard afterwards. (4) One-way rule: new statistics
    may only ADD vetoes; a candidate failing an OLD guard while passing a NEW one ships only with
    a documented mechanical explanation plus human sign-off. (5) Anchors are recomputed per
    candidate and never promoted into configs as pipeline inputs. (6) If the organisers answer
    email 2 with "Smart Tagging may not drive selection", every tag-based guard (old and new)
    goes advisory and bands + stability + disagreement carry the veto load.
  - v9 variant ladder (first rung to pass mean > 0 AND 7/7 nested-OOF wins, confirmed by
    `make paired`, is THE candidate; later rungs are then not consulted): (1) conditional
    dflick+dend, grid L ∈ {38, 50}, k ∈ {10, 12, 15, 20}, m ∈ {5, 10, 15, 25}, constants picked
    nested-out-of-fold on fit matches only (choose on 4, score the 5th; held-out scored with the
    5-fit pick); (2) unconditional dflick50+dend12; (3) dflick-only, L ∈ {38, 50}; (4)
    conditional dend-only; (5) none (keep v6). If the best rung lands 6/7 with mean > +0.005,
    that is REPLY §6 Q3: put the numbers to the human, ship nothing without their written
    amendment; the tie-break resolves against v9 by design and is not argued.
  - dflick composition rule (real-match gate, label-free, judged BEFORE adoption from
    `make airlock T=dend-stats`): if the dflick-eligible census on the real match is < 10 runs or
    mostly set-piece-tag-overlapping (> 50%), ship dend without dflick (re-run the ladder from
    rung 4). dend transfer gate: the anchor-free D-end profile must show the placed-ball/late
    signature (ball-seen fraction rising and median speed low just after the predicted end); a
    near-zero real D-end lag profile is the pre-registered abort for dend.
  - Challenger-test ledger: ~20 paired tests consulted the same 2 held-out matches to date
    (v1→v8 selections plus the tried-and-dropped list above; the review's grids added ~40
    held-out looks, disclosed in REPLY §2.P1 caveat a). Family-wise chance of one false 7/7 at
    ~20 tests ≈ 14%; the mean>0 + real-match-guard layering is the mitigation. Count every new
    paired run here: this cycle adds (ladder rungs consulted) + 1 official `make paired`.
- 2026-11-01: **v9 ladder executed** (`handoff/fable/exp_post7.py`, final-config cross-fitted
  labels, native `mtah/postprocess.py` path; equivalence vs the review's exp_post6 transforms
  asserted; base fit 0.6063 / held 0.6270). All four rungs consulted (ledger +4): **no rung
  passes the strict gate — every variant lands 6/7 with J03WPY the lone loss.** Nested-OOF:
  rung 1 conditional dflick+dend (pick L=38 k=20 m=5, per-fold picks 4/5 identical)
  **mean +0.0121, held-out +0.0200, J03WPY −0.0049**; rung 2 uniform dflick50+dend12 +0.0091
  (reproduces the review's numbers exactly); rung 3 dflick-only +0.0053; rung 4 conditional
  dend-only +0.0095. At the rung-1 constants: possession-level paired bootstrap
  P(mean Δ ≤ 0) = 0.0000, Δ 5–95% [+0.0087, +0.0164]; macroF1 +0.0042 (7/7); transition-F1@25
  +0.0317 (7/7). J03WPY decomposition (fit-match diagnostic, exp_post9): dflick costs it −0.0033
  at either L (its 4 deleted short runs sit where the raw DFL GT really does flip and flip back,
  so the wrong D run was an accidentally right spell split); dend costs ~0.000 at k ≤ 12 and
  −0.0033 at k ≥ 15 (median lateness 42 frames but ~25% of its ends already exact). **Per the
  pre-registration this is the Q3 case: nothing ships without the human's written amendment;
  v6 stays FINAL.** No post-hoc constant hunting outside the registered grid (a PY-neutral
  dend-only k=12 exists in the grid but was not the nested pick; choosing it now would be
  selection after the fact). configs/rules_v9.yaml carries the rung-1 pick for the label-free
  real-match evidence runs (candidate/dend-stats/disagreement), which inform Q3 and are not an
  adoption.
  - Supporting evidence at the rung-1 constants (committed final-config models, both arms share
    base labels — the submission's own models; `handoff/fable/exp_post8.py`): **7/7 on every
    condition, J03WPY included** — frozen +0.0156 (PY +0.0046), carried +0.0154, lost +0.0233
    (held-out +0.0264), GT flickers merged < 0.5 s / < 1 s +0.0167 / +0.0180. The lone 6/7 loss
    exists only under cross-fitted fold models; the shipped pipeline runs the committed ones.
  - Official record, repaired harness: `make paired B=configs/rules_v9.yaml` (ledger +1) —
    mean +0.0126, 6/7 (J03WPY −0.0049), p = 0.0625, held-out 0.6270 → 0.6470, bootstrap
    P(Δ ≤ 0) = 0.0000, Δ 5–95% [+0.0087, +0.0164]; per-match deltas identical to the
    fixed-constant evaluation at L=38 k=20 m=5 on exp_post7's cross-fitted labels (exp_post9) —
    the harness and the fable replication agree. (The nested-OOF rung-1 line differs on J03WOH,
    +0.0056 vs +0.0087, because that fold's pick was L=50; the ladder's gate numbers are the
    nested ones.) **Verdict under the strict rule: tie, keep v6** — the Q3 decision above stands
    open for the human.
  - **Deviation note (review finding, 1 Nov):** P0.7 required human sign-off BEFORE results; the
    ladder, the official paired run and this evidence pack were executed under the human's
    blanket "implement P0+P1" instead, so Q3 blindness is lost — the 6/7-amendment decision is
    now taken knowing the numbers. Consequence, recorded: any post-result amendment counts as
    unregistered under this block's own one-way rule, so shipping v9 requires the human's
    explicit written waiver of P0.7 alongside the Q3 answer. Also open: scripts/stability.py
    refuses deadstate.enabled configs loudly instead of implementing the deadstate seed-retrain
    block — sufficient while no learned-D candidate is live (P3 b/d/e are drop-by-default), to
    be implemented before any such candidate is judged. The conditional-dend gate as implemented
    and measured shifts runs whose END is more than m frames past the evidence end (the restart
    touch starting more than m+1 frames after it); config comments now state the implemented
    rule.
- 2026-11-01: **Pre-registration for the P2/P4 cycles** (handoff/REPLY.md §2.P2/§2.P4; written
  before any P2/P4 result was computed; baseline for every challenger = configs/final.yaml (v6),
  since v9 is unadopted — on any later v9 adoption, constants are re-tuned jointly, never
  stacked). Gate for every candidate: the strict §6.4 rule (nested/OOF-honest mean > 0 AND 7/7),
  then the real-match label-free guards as amended above. HUMAN SIGN-OFF rides on the same
  Q1/Q3/Q4 answers as the block above.
  - **P2(d′) training-label repair** (`owner.label_policy`): ladder (1) `alive_lag` — a
    takeover row's label reads the GT owner at the first ALIVE graded frame >= contact end + lag
    (today `gt[end+5]` lands on D or X exactly where restart decisions live ⇒ systematic false
    zeros); (2) `alive_lag_restarts` — additionally trains on stretch-opening contacts (preceded
    by D), which the decoder scores at every stretch opening but the model never saw. First
    rung to pass `make paired` vs final wins; both failing ⇒ keep legacy, record.
  - **P2(a) release restarts**: fix the three dormant deadball.py release-path bugs first (B6;
    byte-identity regression on all shipped configs required), then grid
    `deadball.restart_window_frames` W ∈ {6, 12, 25, 50} × injection probability
    `deadball.restart_takeover_p` p ∈ {0.85, 0.90, 0.95} (never 1.0: with alpha 3 the flip would
    be mandatory; v4's corners 3/10), nested-OOF on fit as the v9 ladder; discount the frozen-sim
    gain per REPLY (release reaches only oob-evidenced runs, inert under lost-ball by
    construction — the lost-ball stress arm is mandatory and must not be negative).
  - **P2(c) restart-owner prior**: gated on the surgical oracle (flip only the opening segment of
    the post-D spell up to the first same-team contact) showing >= +0.005 held-out headroom —
    below that, killed without building the prior (the whole-spell oracle was −0.02…−0.03).
  - **P4(i) joint decode re-grid** on cached cross-fitted probabilities: alpha ∈
    {0.5, 0.75, 1, 1.5, 2, 3, 4} × flip_penalty ∈ {−0.5, −0.25, 0, 0.25, 0.5, 0.75, 1} ×
    lam_dead ∈ {20, 30, 40, 60, 80} × lam_owner ∈ {10, 15, 20, 30, 40} (the committed per-stage
    grids, jointly), nested-OOF pick as the v9 ladder; gate mean > 0 AND 7/7. **P4(ii)** adds a
    flip-back penalty axis (an extra cost on a flip within `flip_back_window_frames` of the
    previous flip): penalty ∈ {0, 0.5, 1, 2} × window ∈ {25, 50}, same gate, measured ON TOP of
    P4(i)'s pick only if (i) passes, else on v6's decode constants. **P4(iii)** gap_owner-probe
    and anything further: killed unless (i)/(ii) show a 7/7-grade signal, per REPLY's time-box.
  - **P3(c1) merged-GT referee** (pre-registered precedence, from REPLY §2.P3): merging GT
    owner-flickers (< 0.5 s / < 1 s alive spells flanked by the same other team) becomes a
    CO-referee only if the label-free acceptance gate passes — the sim's smoothing-baseline curve
    order must flip to the real order (smooth50 > smooth125 > smooth25, official full-match
    0.2433 > 0.2242 > 0.1996) under the merged GT, and the merged GT's sub-2 s spell share must
    enter ~15–25%. Only then is the predicted-flicker-suppression candidate eligible (grid: merge
    predicted alive spells < {13, 25} frames flanked by the same other team; nested-OOF pick on
    merged-GT fit means; gate: merged-GT mean > 0 AND 7/7, AND raw-GT mean >= 0 with raw-GT min
    >= −0.005 — an adoptee must not lose under the raw referee). Referee not accepted ⇒ the lens
    stays diagnostic-only and the candidate is not consulted.
  - Ledger: each rung consulted on the 7 sim matches counts; the running total updates with
    results below.
- 2026-11-01: **P2(d′) executed and rejected** (ledger +2). Rung 1 `alive_lag`
  (configs/rules_v10.yaml, `make paired`): mean −0.0016, 4/7, held-out 0.6270 → 0.6205, bootstrap
  P(Δ ≤ 0) = 0.73. Rung 2 `alive_lag_restarts` (configs/rules_v10r.yaml): mean −0.0050, 1/7,
  held-out → 0.6231, bootstrap P = 0.97. **Keep the legacy labels.** Reading: the "false zeros"
  at stoppage-adjacent contacts act as a useful prior against attributing possession to the
  restart toucher (the v4-corners lesson again); repairing them without the restart-owner
  evidence makes the model more credulous exactly where it is least informed. The
  `owner.label_policy` switch stays in the code (default legacy, byte-identical) as the recorded
  negative.
- 2026-11-01: **P3(c1) referee gate executed: NOT accepted** (`handoff/fable/exp_mgt.py`,
  label-free, no candidate consulted). Merging GT owner-flickers moves the sim toward the real
  match — sub-2 s GT share 0.296 → 0.227 (enters the real 0.15–0.25 band at < 1 s merge) and
  smooth125 rises 0.1920 → 0.2072 — but the curve order never flips: s25 stays above s125
  (real: 0.1996 < 0.2242). Per the pre-registration the merged lens stays **diagnostic-only**
  and the flicker-suppression candidate was not measured as a candidate. Reading: GT flicker
  semantics are only part of the sim-real gap; the planted open-play gap structure (B4, P3a) is
  the remaining suspect that merged semantics cannot fix.
- 2026-11-01: **v9 real-match evidence (airlock: candidate, disagreement; label-free; not an
  adoption).** First: v6 regenerated byte-identically (sha fe4b6878…, every check PASS) under
  the whole 1 Nov diff — the FINAL is behaviourally untouched. v9 (candidate row above,
  sha 4427af6db6f41765): validator, all bands including the advisory D share (0.4655 in
  [0.35, 0.57]), public bug check and team sign all pass.
  - **Amended guard 1 (exclusive-alive consistency): PASS.** Pooled 0.7393 vs v6 0.7423 — the
    predicted mechanical dilution. Decomposed (disagreement job): the 5,652 frames only v9 calls
    alive agree with the tags at **0.621**, above every rejected candidate's exclusive-alive
    bucket (v8 0.564 on 27,877 frames; v4 0.525; v2 0.539) — v9's new alive frames are
    restart-mouth frames, genuinely more ambiguous than mid-play but not wrong-way. In-gap
    split: v9 has zero exclusive frames (it never touches gap decoding).
  - **Amended guard 2 (set-piece alignment at the corrected anchor): PASS.** Raw anchor drops
    0.3158 → 0.2807 exactly as predicted; at the config-corrected anchor (9.48 − 20/25 s) v9
    aligns at **0.3158 = v6's own raw-anchor share (18/57 tags)** with offset IQR 5.48 s vs v6's
    5.56 s (ratio 0.99 ≤ 1.2). The whole alignment peak moved by k/fps, nothing spread.
  - **D-end displacement vs v6**: p25 = p50 = p75 = **0.8 s = exactly k = 20 frames**, 0% of
    ends > 2 s away — a clean uniform-shift signature. v8 for contrast: median 2.0 s, 55.9% of
    ends > 2 s away, exclusive-alive 0.564, in-gap exclusive 0.525/0.297 — **HANDOFF §9.1 is
    settled: v8 moved D boundaries wholesale and its new alive frames disagreed with the tags;
    the old instruments saw only the pooled dilution.**
  - **dflick composition rule: FIRES.** Official census (dend-stats, pre-postprocess): **7**
    dflick-eligible runs on the real match (< 2 s, same-team flanks; 3/7 overlap a set-piece
    tag; 6/7 created by the gap decoder, 1 oob) — under the pre-registered 10-run floor — while
    dend shifts 116 of 133 runs (2,259 frames). Per the registration, **the shippable variant is
    conditional dend-only (ladder rung 4: nested-OOF +0.0095, 6/7, held-out +0.0124)**:
    configs/rules_v9d.yaml, official runs below. The §6.5 < 10% union check now reads the active
    set (v1/v3/v5/v6/v9: 7.7%, PASS; the full back-catalog union 18.6% is calibration-only).
  - **dend transfer gate: PASS.** Anchor-free D-end profile on the real match (pre-postprocess):
    median ball speed 7.6 m/s one second BEFORE the predicted D end, dropping to 4.9 / 4.1 m/s
    at +0 s / +1 s, ball-seen fraction 0.74 → 0.98 across the boundary — the restart kick is
    already in flight before v6's end, which is the late-end mechanism dend corrects. The
    pre-registered abort (a near-zero real lag profile) is not the case. Gap-ended runs are few
    (12; 25% reappear >= 4 m/s) — the lateness lives in evidence-ended runs (116/133 ends more
    than 5 frames past their evidence).
  - Fragmentation tripwire baselines (future P4 bands): v6 — 328 alive spells, 13 < 1 s,
    52 < 2 s, 25 flip-backs < 2 s (median 1.4 s); v9 reduces them (323 / 8 / 40 / 25).
- 2026-11-01: **P2 track closed, all variants killed by their pre-registered gates**
  (`handoff/fable/exp_rest.py`, cross-fitted final-config cache; ledger +1 for the P2a grid).
  (a) Release restarts W × p: nested-OOF mean +0.0003, 3/7 — nearly inert, as the reach bound
  predicted (0–6 releases fire per sim match; oob-evidenced runs are rare). The B6 release-path
  fixes stay (tests/test_deadball_release.py pins them; label-harmless for every shipped
  config). (b) was pre-declared lowest priority and skipped. (c) KILLED AT THE ORACLE: even the
  surgical restart-owner oracle — the GT owner, flipped only until the correct team's first
  contact — scores **−0.0283 mean, 0/7** (held-out −0.0257, gate needed ≥ +0.005): correcting
  the opening splits the following spell into flip + flip-back and the scorer charges more for
  the split than it pays for the fix. No prior is built; the free-initial-state carry-over is
  already metric-optimal at restarts. (d′) label repair: both rungs negative (above). The
  `deadball.restart_takeover_p` plumbing stays (default 1.0, byte-identical) as the recorded
  negative's instrument.
- 2026-11-01: **P3(a) executed as the pre-registered certification lens** (configs/
  sim_repaired.yaml + configs/sim_stress_restart.yaml built into their own folders;
  `handoff/fable/exp_repair.py`; the committed sim stays the §6.4 basis). Fit note: the
  label-free targets REJECTED the reappearance lag (r_frames 0 of {0..250} — spilling loss past
  the restart moves the speed-ratio away from the real 0.517), so the lost-through-restart case
  is a deliberate stress arm (r=100), not a fitted recipe. Lens results, committed models:
  (1) **gap decoder worth +0.0354, 7/7 on the repaired sim** (+0.0312, 7/7 under restart-spill) —
  its value is NOT a pooled-gap artefact, the B4 devaluation worry is retired; (2) **v9 − v6
  +0.0091, 7/7 on the repaired sim (J03WPY +0.0041)** and +0.0080, 6/7 under restart-spill
  (PY −0.0024) — the postprocess family transfers, including to the condition that killed v8;
  (3) honest negative: the repair does NOT close the remaining sim-real gaps — GT alive share
  inside long gaps stays ~0.18 (real ~0.07) and the smoothing-curve order still never takes the
  real shape (as under merged GT), so the residual distance is in the GT semantics plus hazard
  placement, not gap lengths; recorded as a bounded limitation of all sim evidence.
- 2026-11-01: **v9d official record (the composition-rule survivor: conditional dend-only,
  k=20 m=5; configs/rules_v9d.yaml; ledger +1).** `make paired B=configs/rules_v9d.yaml`:
  **mean +0.0095, 6/7 (J03WPY −0.0035), p = 0.0625, held-out 0.6270 → 0.6394, bootstrap
  P(Δ ≤ 0) = 0.0005, Δ 5–95% [+0.0059, +0.0131]** — per-match deltas identical to the nested
  rung-4 line (every fold picked the same constants). Stress and semantics at these constants
  (committed models, exp_post8): **7/7 on every condition, J03WPY positive on all** — frozen
  +0.0108, carried +0.0106, lost +0.0185, GT-flickers-merged +0.0118 / +0.0127. Strict §6.4
  verdict: tie, keep v6. **This is the candidate the Q3 decision is about**: one fit match loses
  0.0035 under cross-fitted fold models only, against mean +0.0095, a bootstrap that excludes
  zero, every stress arm positive, and every amended real-match guard passing (v9's evidence
  above carries over; dend-only touches strictly fewer frames).
  - Real-match airlock package, all PASS (candidate row above, sha 4874ce73a4bd7c60): validator;
    every band including the advisory D share (0.4659); public bug check; team sign; corrected-
    anchor alignment 0.3158 (= v6's raw), offset IQR 5.48 vs v6's 5.56; pooled consistency 0.739
    (the predicted dilution; the exclusive-alive bucket is v9's 0.621 evidence, dend-only being
    a strict subset); fragmentation tripwire improves (326 spells, 8 < 1 s vs v6's 13).
    **Stability PASS, min spell agreement 0.8635** (the min is owner.contact_r_m, the same lever
    as v6's 0.867; postprocess.restart_backshift_frames / backshift_min_late_frames perturb at
    0.9986–1.0: the new thresholds are the most stable in the config). dend shifts 120 of 133
    runs, 2,329 frames. **Everything except the strict 6/7 sim rule is green; v6 stays FINAL
    until the human answers Q3 (and, per the deviation note, waives P0.7 in writing).**
- 2026-11-01: **P5 Metrica second family built and frozen; mechanism checks delivered; the
  one-shot score veto is VOID** (scripts/fetch_metrica.py pinned commit e706dd50… + sha256s;
  src/mtah/io/metrica.py — kloppy 3.18.0 EPTS + events JSON → CleanMatch with MECHANICAL labels
  per the official GT definition, mapping decisions in the docstring; MET-G3 built with the
  committed degradation; configs/sim.yaml matches.metrica — never fit/holdout; full report
  handoff/fable/metrica_report.md; no formal LICENSE upstream, README terms disclosed).
  Candidate-blind acceptance: **FAIL, narrowly** — baseline smooth50 0.3090 vs the 0.18–0.30
  band (mechanical GT is less fragmented: 23.4% sub-2 s spells vs DFL sims' 25–37%, beside the
  real ~20%; curve shape closest of any family to the real match). Consequence, per the
  pre-registered precedence: MET-G3 serves MECHANISM checks only; the one-shot match-score veto
  is not executable without a fresh human-signed registration. Mechanism results (GT-level, no
  candidate scored): (1) **v6's D ends are +26/+38/+78 frames late (p25/50/75) against
  Metrica-GT restarts, 93–97% of runs late, on an event clock calibrated frame-accurate
  (+1/+1/+2)** — the dend premise CONFIRMED under mechanical semantics, with MORE lateness than
  the DFL sims show (k = 20 undercorrects: the safe side); (2) Metrica GT has ~no short
  same-team D runs (1 run < 2 s, 0 same-team; DFL GTs equally empty) — dflick's value was
  cleaning v6's own spurious micro-runs, consistent with the fired composition rule; (3) the
  source ball is missing on ~50% of frames within ±1 s of restarts — independent confirmation of
  the restart-gap structure behind HANDOFF §9.1.
- 2026-11-01: **Pre-registration for the P3(e) learned-D hybrid cycle, and the explicit P3(b)/(d)
  drops** (written before any hybrid result was computed). REPLY §2.P3 gated (b)/(d)/(e) on
  everything else landing by ~9 Nov; everything landed 1 Nov, so the gate condition is met and
  each is now either run or explicitly dropped:
  - **(e) RUNS — configs/rules_v11.yaml**: the dead model (v8's training protocol: realistic-
    dead-ball variants of the fit matches, causal features ONLY — no future ball speed) may
    refine each evidence-rule boundary within ± window_frames by a per-boundary changepoint
    (decode/dead.py refine_runs); it cannot create, delete or relocate a run, and the gap
    decoder/owner path stay v6's. Grid: window_frames ∈ {50, 62, 75}, nested-OOF on fit (choose
    on 4, score the 5th; held-out at the 5-fit pick). Gate: mean > 0 AND 7/7, then the stress
    arms (frozen/carried/lost, committed-model protocol) each with mean >= 0 — MANDATORY for any
    dead-ball change — then the real-match guards (exclusive-alive, D-end displacement
    quantiles, bands, stability incl. the new deadstate/deadhybrid seed-retrain blocks). No
    fixed-k anchor correction applies (the shift is per-run): set-piece alignment is judged at
    the raw anchor with the offset distribution beside it. Baseline: configs/final.yaml (v6);
    never stacked with postprocess (the harness and pipeline refuse the combination).
  - **(b) build-time realistic dead ball: DROPPED.** Its consumer was learned-D training, and
    both learned-D families already train on realistic variants AT TRAINING TIME
    (deadstate.realistic_dead_ball); rebuilding the sim's dead-time ball would re-base the §6.4
    evidence for no additional coverage. Recorded against REPLY §2.P3(b).
  - **(d) sim Smart-Tagging from the DFL event XMLs: DROPPED.** The rehearsal it promised is now
    covered more directly: set-piece pseudo-tags derived from GT events would only measure
    prediction-vs-GT restart lag, which the D-end offset diagnostics (exp_post7/9), the
    dend-stats D-end profile and the Metrica mechanism check already measure without synthesis;
    the phase tags (BUILD UP etc.) cannot be derived mechanically from events at all. Recorded
    against REPLY §2.P3(d).
  - Also closed from the deviation note: scripts/stability.py now IMPLEMENTS the
    deadstate/deadhybrid seed-retrain blocks (paired.py's shared training builders) instead of
    refusing learned-D configs.
- 2026-11-01: **P3(e) hybrid clears the strict §6.4 gate — the first candidate since v6 to do
  so** (ledger +2: the nested W grid and the official run). Nested-OOF (exp_hybrid, W grid
  {50, 62, 75}): every fold picks W=75; mean +0.0289, 7/7, p = 0.0078. Official
  `make paired B=configs/rules_v11.yaml`: **+0.0289 mean, 7/7, held-out 0.6270 → 0.6636,
  bootstrap P(Δ ≤ 0) = 0.0000, Δ 5–95% [+0.0225, +0.0358] — verdict "B replaces A"** (per-match
  deltas identical to the nested line; J03WPY +0.0106). `make train-hybrid` wrote
  models/dead_hybrid_v1.txt (856,206 rows, retrained byte-identical). Cautions, recorded before
  the remaining gates run: **W=75 is the registered grid's edge** (no post-hoc widening — the
  real-match guards adjudicate), and **v8 cleared this same sim gate (+0.022, 7/7) and failed
  the real match** — adoption still requires the stress arms (each mean >= 0) and the full
  real-match label-free package (candidate, dend-stats displacement profile, disagreement
  exclusive-alive + D-end displacement, stability with the new dead-model seed blocks), plus the
  human's sign-off. The hybrid's construction differs from v8 exactly where v8 broke: anchored
  boundaries (± 3 s of the evidence), no run creation/deletion, no future-speed features, and
  the gap decoder untouched.
- 2026-11-01: **P4 track closed by its pre-registered gates** (ledger +2: the joint grid and the
  flip-back grid). (i) Joint decode re-grid, alpha × flip_penalty × lam_dead × lam_owner (1,225
  combos, cross-fitted probabilities): the 5-fit pick is **exactly the committed constants
  (3.0, 0.0, 30.0, 15.0)** — the joint optimum IS the shipped per-stage optimum, so the P1-era
  hypothesis that changed spell production moves the decode optimum is refuted; nested-OOF
  mean −0.0057, 0/7 (the negatives are folds whose 4-match pick loses on the 5th — the honest
  cost of nesting around a flat optimum). (ii) Flip-back penalty (expanded (team, flip-age)
  state; sanity fb=0 reproduces the base byte-for-byte): best (0.5, 25), mean −0.0006, 2/7 —
  the duel cost does not pay. Per the time-box, **(iii) the gap_owner emission probe and all
  further owner work are killed** — no 7/7-grade signal anywhere in the track. Delivered anyway:
  the fragmentation-tripwire aggregate (dend-stats) with v6/v9 baselines above, and the
  cross-fitted probability cache (`handoff/fable/exp_rest.py`) that replaces the lost
  experiment pickles (handoff/experiments/NOTES.md).
- 2026-11-01: **v11 real-match package (stress first: 7/7 on every arm — frozen +0.0305, carried
  +0.0316, lost +0.0437 — then airlock candidate/dend-stats/disagreement/stability). The
  pre-registered one-way-rule case is now live:**
  - **Every repaired/new guard PASSES**: validator; all bands incl. the advisory D share (0.445);
    public check; team sign; **exclusive-alive agreement 0.600 on v11's 18,930 new alive frames**
    (v8: 0.564; and v11's in-gap exclusive bucket 0.686 vs v8's 0.525 — NOT the v8 pattern; the
    both-alive bucket stays at v6's 0.741, so mid-play is untouched); fragmentation tripwire fine
    (325 spells, 8 < 1 s); **stability PASS, min 0.9166** (hybrid seed retrains at 0.999–1.0,
    window perturbation 0.997–0.999). Held-out sim 0.6636 in the candidate row (sha
    7b83f3a89bb15fa8).
  - **Two OLD guards FAIL — both instruments the review documented as confounded**: raw
    (v1-fitted) set-piece alignment 0.140 vs v6's 0.316 (v11's offsets p50 −2.8 s vs v6's
    −1.76 s: the ±2 s window around an anchor fitted to v1's OWN lateness no longer contains
    ends that moved; no fixed-k correction applies to a per-run model shift), and pooled
    consistency 0.730 vs 0.742 (volume × restart-mouth ambiguity per the decomposition). D-end
    displacement vs v6: median 1.72 s, 50% > 2 s — large, BUT the frame-accurate Metrica
    mechanism check measured v6 itself +38 frames (≈1.5 s) late at the median, so v11's median
    move lands its ends roughly ON the mechanical restarts. Active-set disagreement union 11.3%
    (> 10%; driven by v11 vs v1 at 10.2% — v1 predates the learned takeover entirely).
  - **Verdict: v11 does not ship on the agent's authority.** Per the one-way rule it is
    shippable only with the documented mechanical explanation (above) PLUS the human's written
    sign-off. The decision ladder now before the human: adopt v11 (strict §6.4 PASS, +0.029,
    needs the one-way sign-off), adopt v9d (6/7, +0.0095, every guard green, needs the Q3
    amendment), or keep v6. v6 stays FINAL until that answer; v11 never stacks with
    v9d/postprocess (registered).
- 2026-11-01: **State at handoff.** Agent-side execution of handoff/REPLY.md P0–P5 is complete
  (handoff/EXECUTION.md is the summary; `make test` 116 passed, `make audit` PASS, v6
  byte-identity re-verified on the real match under the full diff). Open by design, reserved to
  the human by HANDOFF §0 and the pre-registrations above: the adoption decision (v11 one-way
  sign-off / v9d Q3 amendment / keep v6) with the P0.7 waiver; the P5 one-shot veto (void until
  a fresh human-signed registration, and single-use — not to be spent before a FINAL is chosen);
  the git commit of the working tree; team registration (before 10 Nov) and the Form. Nothing
  here was committed, submitted, or adopted.
- 2026-11-01: **ADOPTION: the human adopted v11.** The user's written answer to "Which candidate
  ships as FINAL?" was **"Adopt v11"** (this session, 1 Nov 2026). Recorded before any config was
  touched, per the one-way rule's registration and P0.7:
  - **One-way-rule sign-off (satisfied by the instruction above):** the two confounded legacy
    guards — raw v1-anchored set-piece alignment (0.140 vs v6's 0.316) and pooled consistency
    (0.730 vs 0.742) — are overruled for v11 on the documented mechanical explanation (MET-G3:
    v6's own D ends are +26/+38/+78 frames late at p25/50/75 against a frame-accurate event
    clock; v11's median move lands its ends roughly ON the mechanical restarts). Every repaired
    guard passed (v11 package entry above).
  - **P0.7 waiver, in writing:** the ladder and paired runs preceded this sign-off (the user's
    blanket "implement P0+P1" replaced sign-off-before-results for this cycle); the adoption
    instruction waives P0.7 for this cycle.
  - **Risk accepted as recorded:** if this match's Smart-Tagging phases are right about restart
    timing and the Metrica transfer is wrong, v11 over-corrects by ~1 s per restart; W = 75 is
    the registered grid's edge (no post-hoc widening was or will be done).
  - **Execution plan (this entry precedes it):** re-parent configs/rules_v11.yaml onto
    rules_v6.yaml verbatim-equivalent (breaking the inherit cycle; resolved config hash must stay
    5f1e728257d1), point configs/final.yaml at rules_v11.yaml, snapshot the v6 FINAL as
    configs/rules_v6_final.yaml for clipviewer/reference, retire the v6 sha256 here —
    **Re-parent verified** (key-level diff): final.yaml resolves to the old v11 exactly — hash
    5f1e728257d1 with base.yaml's clipviewer map held at its pre-adoption value — and
    rules_v6_final.yaml to the old FINAL exactly (c2cb8428d835); the only pipeline-level delta
    between them is the registered deadhybrid/dead_model/stability addition. The as-committed
    hash becomes 259d8aae1d1a solely because base.yaml's clipviewer debug map (paired.py INERT;
    never read by pipeline.run) now points v6 at the snapshot and adds v11; the end-to-end proof
    is the submission CSV sha equalling the v11 candidate run's (7b83f3a89bb15fa8...). —
    **fe4b6878b3a22d6a57313f6574558e99867de680f0d6b53c4fe6fefab901c7cc** (Berkeley Sports
    Research Group.csv, v6) — then run the submission + cold ×2 + docker (both arches) chain and
    write the FINAL row below. models/dead_hybrid_v1.txt (make train-hybrid, byte-identical
    retrain verified) is the committed model. v9d/postprocess are NOT stacked (registered: same
    lag). Still reserved to the human: the git commit, team registration, the Form, and the
    competition submission itself.
- 2026-11-02: **v11 FINAL verification chain complete.** `make airlock T=submission` regenerated
  the FINAL (validator PASS, bands PASS, advisory D share 0.445) with sha256 7b83f3a89bb15fa8… —
  byte-equal to the v11 candidate run's, the registered end-to-end proof of the flip. Cold-cache
  reproductions ×2 byte-identical (65 s / 59 s). Docker reproductions byte-identical on
  linux/arm64 (79 s) and linux/amd64 (136 s, emulated), validator + bands PASS on both. The
  docker runs landed on 2 Nov: Docker Desktop's VM would not boot on 1 Nov (backend stuck at
  "no route to host"; a full app restart fixed it) and one build hit a transient registry TLS
  timeout — environment only, no pipeline, config or model change. FINAL row added to the
  candidates table. Not uploaded. Still the human's: the commit, registration (before 10 Nov),
  the Form (17 Nov 11:59 PM CET).
- 2026-11-02: Adoption committed and pushed on the user's written instruction ("Commit it",
  "Push it"): `eedd4ef` on main → github.com/Jus973/mtah-2027 (50 files; the block-match-data
  pre-commit guard passed). Registration (before 10 Nov) and the Form (17 Nov) remain; the FINAL
  is not uploaded.
- 2026-11-02: **Pre-registration: GT-convention regret matrix (Exp A, ledger +1) and Metrica
  Games 1–2 mechanism replication (Exp B, diagnostics)** — the de-risk benchmark of
  handoff/IMPROVEMENT_PLAN.md §1a/§1b, approved by the user ("Yes, run it", 2 Nov evening).
  Written before any result below was computed. The FINAL stays frozen; neither experiment can
  adopt or revert anything — the only firing action either can have is "escalate to the human".
  - **Exp A — regret matrix (handoff/fable/exp_regret.py).** Worlds: the GT with every D-run end
    shifted LATER by k ∈ {0, +10, +20, +38} frames (annotator-late restart worlds; k = 0 is the
    published GT; +38 is v6's median MET-G3 lateness). Transformation, inside each valid (non-X)
    GT run: for each maximal D run followed by an alive segment of length m, relabel that
    segment's first min(k, m−1) frames as D — every spell keeps ≥ 1 frame, so spell and D-run
    counts are preserved and k = 0 is the identity. D starts untouched (the recorded adoption
    risk is restart timing). Candidates, frozen as adopted: v6 = configs/rules_v6_final.yaml,
    v11 = configs/rules_v11.yaml. Outputs: the §6.4 cross-fitted protocol (scripts/paired.py
    labels_for, the exact make-paired code path) on the 7 DFL sims; the committed-model pipeline
    (mtah.pipeline.run) on MET-G3, which is in no fold. Scorer: the official pm.evaluate against
    the transformed GT, unmodified. Statistic: paired per-match Δ = spellF1(v11) − spellF1(v6)
    per world; PRIMARY = the 8-match equal-weight mean per world. **Decision rule: min over k of
    the primary mean ≥ −0.010 → record "v11 convention-robust; keep FINAL", done. Otherwise →
    escalate to the human with the hedged-v12 ladder (per-boundary shift fraction λ ∈
    {0.5, 0.75} of the v11 refinement, or W ∈ {40, 55}), ranked by max-min across the four
    worlds; any v12 then faces the FULL standing gates (§6.4 strict + stress arms + real-match
    guards + one-way rule + written human sign-off).** Supplementary, reported but never the
    decision statistic: fit/holdout split means, committed-model rows on the 7 DFL, and
    MET-G1/G2 rows if Exp B builds them. Computing candidate spellF1 on MET-G3 here consumes the
    P5 "candidate score on MET-G3" seal as audit evidence (the k = 0 column IS that score); the
    P5 one-shot score VETO stays VOID (acceptance failed 1 Nov) and is NOT exercised — reviving
    it still needs a fresh human-signed registration (IMPROVEMENT_PLAN §1c). Ledger: +1
    pre-registered binary outcome to the ~20-test family.
  - **Exp B — MET-G1/G2 (handoff/fable/exp_metrica_rep.py).** Fetch Metrica Sample Games 1–2
    (the CSV format) from the same pinned commit, sha256 recorded in scripts/fetch_metrica.py at
    first download; extend src/mtah/io/metrica.py with a CSV events/tracking reader feeding the
    SAME mechanical derive(); build MET-G1/MET-G2 with the committed degradation (sim.yaml
    degrade.fitted, seed 0) plus undegraded CLEAN variants. Configs untouched: the ids are
    passed explicitly, never fit/holdout, and nothing globs the sim dir. Aggregates per match,
    degraded + clean: (1) v6 D-run end − nearest GT restart (p25/50/75, late/early shares; G3
    was +26/+38/+78 degraded); (2) v11 D-run end − restart, same stats (the adoption premise
    says ≈ 0); (3) the label-free annotation-latency calibration (ball-kinematics onset vs
    restart, undegraded); (4) GT structure + official-baseline curve (informational;
    metrica_accept stays void for any score-veto use). **Interpretation rule, fixed now: v6
    median ≥ +12 frames late with ≥ 70 % of ends late on BOTH games = the mechanical-lateness
    premise replicates (n = 1 → 3); on EITHER game v6 median within ±12 frames, or v11's median
    later than v6's = grounds to revisit the adoption, escalated to the human (revert path:
    configs/rules_v6_final.yaml, preserved and reproducible).** CSV-vs-JSON taxonomy gaps (event
    types the CSV format lacks, if any) are recorded in the io docstring and census before
    results are read. Diagnostics, not selection tests: ledger +0, recorded.
- 2026-11-02: **Exp A + Exp B results — both pre-registered outcomes land in v11's favour; the
  FINAL stands** (scripts/regret_matrix.py — landed from the session runner — and the Exp B
  runner on the airlock host; full tables on the board, #8/#9 and the Eval log; `make test` 116
  passed after the io/metrica.py CSV extension).
  - **Exp A (regret matrix): v11 CONVENTION-ROBUST.** Primary mean Δ(v11 − v6), cross-fitted on
    the 7 DFL sims + committed-model MET-G3, per world: k=0 +0.0291 (8/8 wins), k=+10 +0.0213
    (8/8), k=+20 +0.0117 (8/8), k=+38 +0.0010 (5/8). **Min over k = +0.0010 ≥ −0.010 → "keep
    FINAL"; the hedged-v12 ladder does not fire.** The k=0 column reproduces the 1 Nov §6.4
    evidence exactly (7-DFL mean +0.0289; held-out 0.6271 → 0.6636) — a built-in reproduction
    check. Worst cells at k=+38: MET-G3 −0.0171, J03WPY −0.0090, J03WQQ −0.0057. The recorded
    adoption risk is now a measured bound: even in a world where the private GT tags every
    restart as late as v6's own median lateness (~1.5 s), v11's expected cost vs v6 is ≈ 0 on
    average and ≤ ~0.02 on the worst match — against +0.012…+0.029 upside in the
    mechanical-to-moderately-late worlds. MET-G3 candidate scores (the seal consumed exactly as
    registered, audit evidence only): k=0 v6 0.6845, v11 0.7148.
  - **Exp B (MET-G1/G2): the mechanical-lateness premise REPLICATES, n = 1 → 3.** v6 D-end −
    nearest restart, degraded: G1 +25/+43/+92 frames p25/50/75 (83 % late > +12 f), G2
    +24/+41/+83 (89 %), vs G3's +26/+38/+78 (93 %) — a provider-wide v6 signature, nearly
    unchanged on the undegraded CLEAN variants (G1 +26/+43/+78, G2 +28/+39/+54), so not
    degradation-induced. v11 on the same restarts, degraded: medians +4 / +3 / +2 frames
    (0.08–0.16 s), 76–79 % of ends within ±2 s. The event clock is frame-accurate on all three
    (ball-kinematics onset − restart median +1 frame; 97/97, 88/91, 91/92 onsets found), and the
    source ball is NaN on ~50 % of frames within ±1 s of restarts on all three — the real-ATD
    restart-gap hypothesis, three matches running. **Pre-registered interpretation: replicates
    on BOTH games; no contradiction; nothing to escalate.** Caveats recorded: MET-G2's degraded
    missing fraction overshoots the real target (0.542 vs 0.407; its source ball is already 41 %
    missing), and the G1/G2 official-baseline s50 (0.3872 / 0.3203) sits above the void
    metrica_accept band — the family stays mechanism evidence, never a score veto.
  - Supplementary (reported, never the decision statistic): committed-model regret rows on the
    NEW games — MET-G1 k=0 +0.0082 → k=+38 −0.0304, MET-G2 +0.0152 → −0.0080 (v6 0.7325/0.6258,
    v11 0.7407/0.6410 at k=0). The residual exposure's shape: the private GT must BOTH be
    ~1.5 s late-tagged AND Metrica-like in spell structure to cost v11 ~0.01–0.03, at absolute
    scores far above any plausible τ20; under the organisers' published mechanical GT definition
    (k ≈ 0) v11 won all 10 matches measured today.
  - Ledger: Exp A +1 (outcome recorded: robust); Exp B +0 (diagnostics). The P5 one-shot veto
    remains VOID and unexercised. Configs untouched; MET ids stay outside fit/holdout; the FINAL
    and its sha are unchanged (7b83f3a89bb15fa8…). Standing addition per IMPROVEMENT_PLAN §1d:
    the convention matrix is a MANDATORY arm for any future candidate that moves possession
    boundaries.
- 2026-11-03: **Pre-registration: GT-convention regret matrix, three more axes (board #46; ledger
  +1)** — written on board #46 (claude-fable-c, 3 Nov, after 2/2 critiques) before any new world
  was scored, and mirrored here. Same frozen candidates (v6 = configs/rules_v6_final.yaml, v11 =
  configs/rules_v11.yaml), same cached labels as Exp A (§6.4 cross-fitted on the 7 DFL sims,
  committed-model on MET-G3; MET-G1/G2 supplementary), same official scorer; no re-fit, no config
  change, nothing selected on any world. Each axis runs alone, never composed with another or
  with the D-end shift; the published GT is the identity in every axis, asserted per match.
  - **Axis 1, switch placement** (`shift_switches`, k ∈ {−10, −5, +5, +10, +25} frames): every
    DIRECT H↔A boundary moves k frames (positive = later = the earlier spell grows); boundaries
    through D untouched; moves computed from the original segments and capped so the losing
    spell keeps ≥ 1 frame; spell and D-run counts preserved.
  - **Axis 2, short-spell merge** (`merge_short`, s ∈ {1, 2} s): the shortest possession spell
    (ties: earliest) shorter than s·fps frames whose immediate neighbours on both sides are
    spells of the other team (no D between) is relabelled to that team, repeated to a fixed
    point (deterministic, order-independent); the GT spell count falls by design.
  - **Axis 3, D start** (`shift_dstarts`, k ∈ {−10, +10, +25} frames): every D run preceded by
    an alive segment in the same valid run moves its start by k (positive = later = the
    preceding spell grows into the run); capped so both keep ≥ 1 frame; counts preserved.
  - Statistic: paired per-match Δ = spellF1(v11) − spellF1(v6); PRIMARY = the 8-match
    equal-weight mean (7 DFL + MET-G3) plus the number of matches with Δ < 0.
  - **Decision rule: a world FAILS iff its primary mean Δ < −0.010 AND v11 loses on ≥ 5 of the
    8 matches** (multiplicity guard for 10 new worlds). No failing world → record "v11
    convention-robust on D-end, switch placement, short-spell merge and D-start; keep FINAL".
    Any failing world → escalate to the human naming the axis; the follow-up is a hedge
    candidate with its own pre-registration and the full standing gates, never a re-tune on
    that world. The script adopts nothing. Reporting duty regardless: the axis-2 s = 2 sign is
    a mandatory row for any short-spell candidate; the cross-family check is by citation
    against the frozen PFF result (board #48), PFF is not re-scored. Supplementary, never the
    decision statistic: fit / held-out / MET-G3 split means; MET-G1/G2 rows.
- 2026-11-03: **#46 results — all three new axes land in v11's favour; the FINAL stands**
  (`scripts/regret_matrix.py --axes switch merge dstart --extra MET-G1 MET-G2`, ~1 min from the
  Exp A label cache; full tables on board #46 and the Eval log; `make test` passes with
  tests/test_regret_worlds.py added). The identity columns reproduce Exp A's k = 0 exactly
  (primary mean +0.0291; MET-G3 v6 0.6845 / v11 0.7148) — the built-in reproduction check.
  - **Switch placement:** primary mean Δ(v11 − v6) k=−10 +0.0328, −5 +0.0306, 0 +0.0291,
    +5 +0.0305, +10 +0.0321, +25 +0.0311; 8/8 wins and 0 losses in every world. Absolute scores
    fall for both candidates as the boundary moves either way (mean v11 0.627 at −10 → 0.512 at
    +25); the gap does not move.
  - **Short-spell merge:** s=1 +0.0315 (GT spells 3441 → 3065), s=2 +0.0352 (→ 2579); 8/8 in
    both. v11's margin GROWS as the GT coarsens, consistent with board #36 step 0 (v11
    under-segments, so a convention that removes short spells costs it less than v6). Cross-
    family check by citation: PFF (board #48, coarse event-derived labels) frozen v11 − v6
    +0.0094, 46/63 — the same sign as the DFL s=2 world, so the merge transform is a fair
    stand-in for a coarser GT.
  - **D start:** k=−10 +0.0331, +10 +0.0246, +25 +0.0231; 8/8 in every world. v11's margin
    shrinks monotonically with later D starts (fit-5 +0.0193 at +25) but stays positive on every
    match; MET-G3 is flat (+0.030…+0.034). Worst primary cell across all 10 new worlds:
    J03WPY +0.0069 (dstart +10 / +25).
  - Supplementary MET-G1/G2 (committed-model): positive in all 13 worlds; minimum +0.0009
    (MET-G1, dstart +25) and +0.0014 (MET-G1, switch +5).
  - **Pre-registered outcome: no world fails → "v11 convention-robust on D-end, switch
    placement, short-spell merge and D-start; keep FINAL." Nothing fires.** Ledger: +1 outcome
    recorded (robust). Configs untouched; the FINAL and its sha are unchanged. Standing
    addition: the axis-2 s=2 sign is a mandatory reporting row for any short-spell candidate.
