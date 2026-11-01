# Submission log

One row per candidate CSV (PLAN.md §6.6). The final choice is the row marked FINAL.

| datetime | candidate | config sha | git SHA | csv sha256 | public check (verdict only) | held-out sim spellF1 | bands | notes | uploaded? |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-31 18:10 | v1 | f96f2f64a285 | 4cbeed1 | 3c32c4758d117a6a | pass | 0.5406 | pass | consistency 0.740 | no |
| 2026-10-31 18:10 | v1-h2 | bc6afc21311e | 4cbeed1+dirty | 6b2c1299a6e2a7e2 | pass | 0.5217 | FAIL | consistency 0.714 | no |
| 2026-10-31 20:30 | v2 | 14177d87de60 | ef58ae0 | a5a8eb85099c74fb | pass | 0.5450 | pass | consistency 0.728; near-tie with v1, not selected (§6.4) | no |
| 2026-10-31 20:46 | v3 | 2cb112d12c53 | 6dfcf31 | 00f4551f83d84c89 | pass | 0.5838 | pass | consistency 0.740; leading candidate, 7/7 sim matches beat v1; selection pending (§6.4) | no |


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
