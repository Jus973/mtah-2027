# MTAH 2027: ball possession from automatic tracking

Our entry for More Than A Hack 2027, Phase 1 (FC Barcelona vs Athletic Club). The output is a
per-frame `H / A / D` possession file for the tactical video. `PLAN.md` is the spec and
`PLAN_STATUS.md` is the running state.

**No match data is in this repository.** The gated bundle stays wherever `MTAH_DATA_DIR` points.
Derived caches go to `${MTAH_DATA_DIR}_cache` (override with `MTAH_CACHE_DIR`).

```bash
make setup                                  # venv (Python 3.11), pinned deps, pre-commit hook
make test                                   # runs with NO real data (synthetic match + public GT)
make fake                                   # write a synthetic match to fakedata/out/BAR-FAKE

export MTAH_DATA_DIR=/path/to/mtah2027      # real data: humans run these, paste back aggregates
make cache                                  # parse ATD -> npz cache, check published counts
make baseline-check                         # regenerate official baselines byte-for-byte
make anchors                                # derive kick-offs + attacking direction, assert them
make atd-stats                              # label-free degradation statistics (sim targets)
make airlock T=<target>                     # how an agent runs any of these (aggregates only)

make open-data                              # DFL/IDSSE open dataset, 2.63 GB, CC BY -> ~/opendata/idsse
make sim-fit                                # fit degradation parameters to the real ATD statistics
make sim                                    # degraded open-data matches -> ~/opendata/sim/DFL-*
make sim-check                              # sim acceptance test: official baseline ~0.24 on the sim
make train-takeover                         # v3 takeover model from the sim fit matches -> models/

make doc TEAM=<TEAM_NAME> REPO=<url>        # technical document -> docs/<TEAM_NAME>.md (+ .pdf)
```

Simulation data: Bassek, M., Rein, R., Weber, H. et al. *An integrated dataset of spatiotemporal
and event data in elite soccer.* Scientific Data 12, 195 (2025),
doi:10.6084/m9.figshare.28196177 (CC BY 4.0). Degraded copies are rebuilt by `make sim`, never
committed. `models/takeover_v1.txt` is a LightGBM model trained on the degraded sim only (no match
data); `make train-takeover` regenerates it byte-identically.

Layout: `src/mtah/` (pipeline), `configs/` (every threshold), `fakedata/` (synthetic generator),
`scripts/` (make-target entry points), `evaluation/`, `starter/`, `games.py` (vendored official
code, unmodified; checked by `tests/test_hygiene.py`).

macOS note: LightGBM needs `brew install libomp`.
