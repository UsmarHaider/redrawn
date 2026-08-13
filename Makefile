# redrawn -- reproduce every number in the README from the raw open data.
#
#   make all      download, build, analyse, plot, export      (~25 min)
#   make test     the offline test suite                      (no data needed)
#   make serve    the web interface on :8000

PY := .venv/bin/python
PIP := .venv/bin/pip
PORT ?= 8000
REPLICATES ?= 40
STEPS ?= 800000

.PHONY: all venv data build describe sweep gerrymander convergence inference \
        analysis figures web test serve screenshot summary clean distclean

all: data build analysis figures web summary

venv:
	python3 -m venv .venv
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"

## Fetch 2.3M crash records and four boundary layers from NYC Open Data (~50 MB).
## Never downloaded in CI: every test builds its own synthetic city.
data:
	$(PY) scripts/download_data.py

## Clean, geocode against all four layers, cache the tables. The slow step:
## two million point-in-polygon tests, about eight minutes.
build:
	$(PY) -m redrawn.cli build

describe:
	$(PY) -m redrawn.cli describe

sweep:
	$(PY) -m redrawn.cli sweep --replicates $(REPLICATES)

gerrymander:
	$(PY) -m redrawn.cli gerrymander --steps $(STEPS)

convergence:
	$(PY) -m redrawn.cli convergence

inference:
	$(PY) -m redrawn.cli inference

analysis: describe sweep gerrymander convergence inference

figures:
	$(PY) -m redrawn.cli figures

web:
	$(PY) -m redrawn.cli web

summary:
	$(PY) -m redrawn.cli summary

test:
	$(PY) -m pytest

serve:
	$(PY) -m uvicorn "redrawn.web:get_app" --factory --port $(PORT) --reload

## Capture the README screenshot from the running service (macOS Chrome path).
screenshot:
	$(PY) scripts/screenshot.py --port $(PORT)

clean:
	rm -rf artifacts reports docs/figures src/redrawn/ui/data.json

distclean: clean
	rm -rf data .venv .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
