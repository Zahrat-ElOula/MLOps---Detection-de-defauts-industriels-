# ---------------------------------------------------------------------------
# industrial-defect-detection - developer commands
# Works on Linux/macOS and Windows (Git Bash / MSYS make).
# ---------------------------------------------------------------------------
VENV ?= .venv
PYTHON_BOOTSTRAP ?= python3.11
PYTHON ?= $(if $(wildcard $(VENV)/Scripts/python.exe),$(VENV)/Scripts/python.exe,$(VENV)/bin/python)
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/params.yaml

.PHONY: help install train test api frontend docker-build docker-up \
        dvc-init dvc-pull clean

help:
	@echo install      Create the virtualenv and install dependencies
	@echo train        Run the reproducible training pipeline
	@echo test         Run the test suite
	@echo api          Serve the FastAPI application on port 8000
	@echo frontend     Start the Vite development server
	@echo docker-build Build API and frontend images
	@echo docker-up    Start the local stack with docker compose
	@echo dvc-init     Initialize DVC (idempotent)
	@echo dvc-pull     Pull DVC-tracked data
	@echo clean        Remove caches and build artefacts

# Windows: make install PYTHON_BOOTSTRAP="py -3.11"
install:
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) pipelines/train_pipeline.py --params $(CONFIG)

test:
	$(PYTHON) -m pytest tests -v

api:
	$(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd frontend && npm run dev

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

dvc-init:
	$(PYTHON) -m dvc init --force
	$(PYTHON) -m dvc add data/raw || true

dvc-pull:
	$(PYTHON) -m dvc pull

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache', 'htmlcov', 'dist', 'build', 'frontend/dist', 'frontend/node_modules')]"
