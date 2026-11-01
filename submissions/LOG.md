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
| 2026-11-01 10:16 | final | 85a3d741a110 | 9e6206d | fe4b6878b3a22d6a | pass | 0.6270 | pass | **FINAL** (§6.4): configs/final.yaml = v6, same file as v6; consistency 0.742 | no |


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
