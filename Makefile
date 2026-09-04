.PHONY: install install-dev download-data train-baseline train train-all tune evaluate serve test lint docker-build docker-run monitor-drift clean

PYTHON ?= python
CONFIG ?= configs/train.yaml
MODEL ?= artifacts/model.joblib

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,lightgbm,eda]"

download-data:
	$(PYTHON) -m taxi_duration.cli download-data --config $(CONFIG)

train-baseline:
	$(PYTHON) -m taxi_duration.cli train --config $(CONFIG) --models baseline

train:
	$(PYTHON) -m taxi_duration.cli train --config $(CONFIG)

train-all:
	$(PYTHON) -m taxi_duration.cli train --config $(CONFIG) --models baseline ridge hist_gbr lightgbm

tune:
	$(PYTHON) -m taxi_duration.cli tune --config $(CONFIG)

evaluate:
	$(PYTHON) -m taxi_duration.cli evaluate --config $(CONFIG) --model-path $(MODEL)

serve:
	uvicorn taxi_duration.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

lint:
	ruff check src tests

docker-build:
	docker build -t taxi-duration-service .

docker-run:
	docker run --rm -p 8000:8000 -v "%cd%/artifacts:/app/artifacts" taxi-duration-service

monitor-drift:
	$(PYTHON) -m taxi_duration.cli monitor-drift --config $(CONFIG)

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', 'reports', 'mlruns']]"
