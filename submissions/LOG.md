# Submission log

One row per candidate CSV (PLAN.md §6.6). The final choice is the row marked FINAL.

| datetime (PT) | candidate | config sha | git SHA | csv sha256 | public spellF1 / macroF1 | sim spellF1 | bands | notes | uploaded? |
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
