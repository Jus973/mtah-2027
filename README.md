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
```

Layout: `src/mtah/` (pipeline), `configs/` (every threshold), `fakedata/` (synthetic generator),
`scripts/` (make-target entry points), `evaluation/`, `starter/`, `games.py` (vendored official
code, unmodified; checked by `tests/test_hygiene.py`).

macOS note: LightGBM needs `brew install libomp`.
