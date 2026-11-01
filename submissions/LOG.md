# Submission log

One row per candidate CSV (PLAN.md §6.6). The final choice is the row marked FINAL.

| datetime | candidate | config sha | git SHA | csv sha256 | public check (verdict only) | held-out sim spellF1 | bands | notes | uploaded? |
|---|---|---|---|---|---|---|---|---|---|

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
