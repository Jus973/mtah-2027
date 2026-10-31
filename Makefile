# MTAH 2027. Real-data targets need MTAH_DATA_DIR and print aggregates only (CLAUDE.md).
# Humans run them in a terminal; agents run them through the airlock: make airlock T=<target>.
PY      ?= .venv/bin/python
GAME    ?= BAR-ATH
FAKE    ?= BAR-FAKE
export PYTHONHASHSEED := 0

.PHONY: setup fake test airlock cache baseline-check anchors anchors-diag probe sim-costs score autopsy candidate sync submission doc check-data

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps
	.venv/bin/pre-commit install

fake:
	$(PY) fakedata/generate.py --game $(FAKE) --seed 0 --half-s 900

test:
	$(PY) -m pytest

# make airlock T=anchors: a real-data target with validated, aggregate-only output (scripts/airlock.py).
# The bundle path comes from MTAH_DATA_DIR or .mtah.local; the full log stays beside the cache.
T ?=
airlock:
	@$(PY) scripts/airlock.py $(T) --game $(GAME)

check-data:
	@test -n "$(MTAH_DATA_DIR)" || (echo "MTAH_DATA_DIR is not set" && exit 1)

cache: check-data
	$(PY) scripts/run_cache.py --game $(GAME)

baseline-check: check-data
	$(PY) scripts/baseline_check.py --game $(GAME)

anchors: check-data
	$(PY) scripts/run_anchors.py --game $(GAME)

anchors-diag: check-data
	$(PY) scripts/diag_kickoff.py --game $(GAME)

probe: check-data
	$(PY) scripts/probe_streams.py --game $(GAME)

sim-costs:
	$(PY) scripts/sim_costs.py

# make score CSV=path/to.csv GT=public|sim:<game>|path
GT ?= public
score:
	$(PY) -c "import json,sys; from mtah.eval.score import score; print(json.dumps(score('$(CSV)', '$(GT)', bootstrap=1000), indent=2))"

autopsy candidate sync submission doc:
	@echo "make $@: not implemented yet (PLAN.md M1+). See PLAN_STATUS.md." && exit 1
