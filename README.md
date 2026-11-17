# MTAH 2027: ball possession from automatic tracking

Our entry for More Than A Hack 2027, Phase 1 (FC Barcelona vs Athletic Club). The output is a
per-frame `H / A / D` possession file for the tactical video. `PLAN.md` is the spec,
`PLAN_STATUS.md` the running state, `submissions/LOG.md` every candidate and decision, and
`docs/technical.md` the technical document.

**No match data is in this repository.** The gated bundle stays wherever `MTAH_DATA_DIR` points.
Derived caches go to `${MTAH_DATA_DIR}_cache` (override with `MTAH_CACHE_DIR`).

## Reproduce the submission

```bash
make setup                                  # venv (Python 3.11), pinned deps, pre-commit hook
make test                                   # runs with NO real data (synthetic match + public GT)
export MTAH_DATA_DIR=/path/to/mtah2027      # the bundle: tactical/ and broadcast/
make cache                                  # parse the ATD into an npz cache outside the repo
make submission                             # -> "submissions/final/Berkeley Sports Research Group.csv"
```

`make submission` runs `configs/final.yaml` (the selected candidate, v12-hedgeA-seq — adopted 16 Nov on
the decision recorded in `submissions/LOG.md` after an independent verification), the official validator
and the structural bands on the exact file. It records the file's sha256 in
`submissions/final/<team>.sha256`, or checks it against that file when it exists. The team name
(`team` in `configs/base.yaml`; override with `TEAM="..."`) is the file name. The
submitted file has sha256 `077e0932a503e1b78f8f0053c51f75ac635939233180f91f07fc9140f0677383`
(`submissions/final/Berkeley Sports Research Group.sha256`). Runs are deterministic: CPU only, no randomness at
inference, single-threaded LightGBM, `PYTHONHASHSEED=0`. About 30 s once the cache exists.

With Docker (the real bundle opens only at a terminal, hence `-t`). The image gives the same
file on linux/arm64 and linux/amd64:

```bash
make docker-build                           # docker build -t mtah . (PLATFORM=linux/amd64 for x86)
docker run -t -v /path/to/mtah2027:/data -v "$PWD/out":/out mtah
```

## How it works

Every contact between the ball and a player is a candidate possession change. A LightGBM model
trained only on degraded open data (`models/takeover_v2.txt`) gives each contact by the
non-owning team a takeover probability, from the contact's kinematics and both teams' shape. A two-state Viterbi over the contacts
(`src/mtah/decode/viterbi.py`) picks the owner path. Dead-ball stretches come from the tracking:
the ball out of bounds or lost by the tracker, up to the restart (`src/mtah/events/deadball.py`).
Inside long ball gaps, two more sim-trained models read the players alone (team shape and motion,
`models/gap_*_v1.txt`), and a three-state Viterbi re-decides `H`, `A` or `D`
(`src/mtah/decode/gaps.py`). See `docs/technical.md`.

## Selection and checks (no labels on this match)

```bash
make open-data                              # DFL/IDSSE open dataset, 2.63 GB, CC BY -> ~/opendata/idsse
make sim                                    # degraded open-data matches -> ~/opendata/sim/DFL-*
make sim-check                              # sim acceptance test: official baseline ~0.24 on the sim
make train-takeover                         # retrain models/takeover_v1.txt (byte-identical)
make train-gaps                             # retrain the gap models (byte-identical) + paired test
make sim-ablate                             # held-out ablations of the final config
make candidate NAME=v6 CFG=configs/rules_v6.yaml   # a candidate + bands, consistency, public check
make paired A=configs/rules_v5.yaml B=configs/rules_v6.yaml   # §6.4 paired test, cross-fitted
```

Agents run real-data targets only through the airlock, which releases policy-checked aggregates:
`make airlock T=<target>` with `cache`, `candidate`, `submission`, `stability`, `disagreement`,
`dead-stats`, `public-check`, `audio`, `sync`, `whistle` and others (see `scripts/airlock.py`).
Humans only: `make autopsy`, and `make clips` for `tools/clipviewer/index.html`. Both are for
debugging; they never produce labels.

```bash
make audit                                  # repo audit: no bundle files, sizes, secrets
make doc                                    # writes docs/<team>.md and docs/<team>.pdf
```

The document build requires Pandoc, XeLaTeX, Poppler’s `pdfinfo`, and the original STIX
font family (`STIXGeneral`) installed for XeLaTeX. On Debian/Ubuntu the packages are
`pandoc texlive-xetex fonts-stix poppler-utils`. Set `REPO=<url>` to override the default
repository link. A build over the six-page limit fails.

## Data and licences

Simulation data: Bassek, M., Rein, R., Weber, H. et al. *An integrated dataset of spatiotemporal
and event data in elite soccer.* Scientific Data 12, 195 (2025),
doi:10.6084/m9.figshare.28196177 (CC BY 4.0). Degraded copies are rebuilt by `make sim` and never
committed. Every committed model (`models/takeover_v1.txt`, `takeover_v2.txt`, the two
`gap_*_v1.txt`, `dead_hybrid_v1.txt` — the boundary model — and `proxowner_v1.txt` / `proxowner_v2.txt` —
the proximity owner models, v2 the final's — and `seqnet_v1.json`, the final's sequence owner model) is trained on that sim only.

Layout: `src/mtah/` (pipeline), `configs/` (every threshold), `models/`, `fakedata/` (synthetic
generator), `scripts/` (make-target entry points), `tools/clipviewer/`, plus `evaluation/`,
`starter/` and `games.py`, the vendored official code, unmodified and checked by
`tests/test_hygiene.py`.

macOS note: LightGBM needs `brew install libomp`.
