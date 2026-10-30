# Submission log

One row per candidate CSV (PLAN.md §6.6). The final choice is the row marked FINAL.

| datetime (PT) | candidate | config sha | git SHA | csv sha256 | public spellF1 / macroF1 | sim spellF1 | bands | notes | uploaded? |
|---|---|---|---|---|---|---|---|---|---|

## Work log

- 2026-10-30: M0 scaffold. Vendored official repo @ f8f2629. `scripts/sim_costs.py` reproduces
  the PLAN §2 cost table exactly on the public GT. Fake match generator, ATD loader, kick-off and
  direction detection, and 17 tests, all passing with no real data present. Real-data checks are
  pending a human run (see PLAN_STATUS.md).
