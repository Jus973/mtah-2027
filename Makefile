# MTAH 2027. Real-data targets need MTAH_DATA_DIR and print aggregates only (CLAUDE.md).
# Humans run them in a terminal; agents run them through the airlock: make airlock T=<target>.
PY      ?= .venv/bin/python
GAME    ?= BAR-ATH
FAKE    ?= BAR-FAKE
export PYTHONHASHSEED := 0

.PHONY: setup fake test airlock cache baseline-check anchors anchors-diag probe atd-stats open-data sim-fit sim sim-check sim-rules sim-dead train-takeover sim-latency sim-costs score autopsy candidate sync submission doc check-data

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
	@$(PY) scripts/airlock.py $(T) --game $(GAME) $(if $(CSV),--csv $(CSV)) $(if $(CFG),--config $(CFG)) $(if $(NAME),--name $(NAME))

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

atd-stats: check-data
	$(PY) scripts/atd_stats.py --game $(GAME)

# DFL/IDSSE open data (2.63 GB, CC BY) to $$MTAH_OPEN_DIR (default ~/opendata/idsse).
open-data:
	$(PY) scripts/fetch_open_data.py

# Sim on degraded open data (PLAN.md §6.1): fit degradation, build DFL-* bundles, acceptance test.
sim-fit:
	$(PY) scripts/fit_sim.py

sim:
	$(PY) scripts/build_sim.py

sim-check:
	$(PY) scripts/sim_check.py

# Tune owner hypotheses H1-H3 on the sim's fit matches, compare on held-out (D = sim GT oracle).
sim-rules:
	$(PY) scripts/sim_rules.py

# Tune dead-ball v1 on the sim's fit matches; ablate and compare on held-out.
sim-dead:
	$(PY) scripts/sim_deadball.py --config $(or $(CFG),configs/rules.yaml)

# Takeover model for the v3 decoder: trained on the sim's fit matches only -> models/ (committed)
train-takeover:
	$(PY) scripts/train_takeover.py --config $(or $(CFG),configs/rules_v3.yaml)

# Transition timing residuals and latency corrections, calibrated on fit, judged on held-out
sim-latency:
	$(PY) scripts/sim_latency.py --config $(or $(CFG),configs/final.yaml)

sim-costs:
	$(PY) scripts/sim_costs.py

# make score CSV=path/to.csv GT=sim:<game>|path. The public window refuses: candidates get a
# pass/fail verdict only, via make airlock T=public-check CSV=... (RULES.md: not a tuning set).
GT ?=
score:
	$(PY) -c "import json,sys; from mtah.eval.score import score; print(json.dumps(score('$(CSV)', '$(GT)', bootstrap=1000), indent=2))"

# make candidate CFG=configs/rules.yaml NAME=v1 (humans; agents use make airlock T=candidate ...)
candidate: check-data
	$(PY) scripts/run_candidate.py --game $(GAME) --config $(or $(CFG),configs/rules.yaml) --name $(NAME)

# make doc TEAM=<TEAM_NAME> REPO=<public repo URL> -> docs/<TEAM_NAME>.md (+ .pdf with pandoc)
doc:
	$(PY) scripts/build_doc.py --team "$(TEAM)" --repo "$(REPO)"

# make submission TEAM=<TEAM_NAME> (agents: make airlock T=submission NAME=<TEAM_NAME>)
submission: check-data
	$(PY) scripts/make_submission.py --game $(GAME) --config $(or $(CFG),configs/final.yaml) --name $(or $(TEAM),TEAM_NAME)

autopsy sync:
	@echo "make $@: not implemented yet (PLAN.md M1+). See PLAN_STATUS.md." && exit 1
