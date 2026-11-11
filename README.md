<div align="center">

# 🏭 Industrial Defect Detection

**End-to-end defect detection on the MVTec AD dataset, combining a Deep Learning module (PyTorch / YOLO) with a complete MLOps stack.**

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00B8D4?style=flat-square&logo=ultralytics&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Inference%20API-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?style=flat-square&logo=react&logoColor=black)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?style=flat-square&logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containers-2496ED?style=flat-square&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=flat-square&logo=github-actions&logoColor=white)
![CI](https://github.com/Zahrat-ElOula/MLOps---Detection-de-defauts-industriels-/actions/workflows/ci.yml/badge.svg?branch=main)
![CT](https://github.com/Zahrat-ElOula/MLOps---Detection-de-defauts-industriels-/actions/workflows/ct-retrain.yml/badge.svg?branch=main)
![CD](https://github.com/Zahrat-ElOula/MLOps---Detection-de-defauts-industriels-/actions/workflows/cd.yml/badge.svg?branch=main)

<br>

[📋 Overview](#-overview) •
[🏗 Architecture](#-architecture) •
[✨ Features](#-features) •
[📦 Getting Started](#-getting-started) •
[🚀 Usage](#-usage) •
[🧠 Deep Learning](#-deep-learning-pipeline) •
[⚙️ MLOps](#️-mlops-pipeline)

</div>

---

> [!NOTE]
> **Status: Completed.** The system is fully functional end to end, featuring a two-stage architecture (multi-class classification + detection), a containerized API, a modern web interface, and a complete CI/CD/CT MLOps pipeline. The Continuous Training workflow is triggered by a push on `data/feedback/`, a weekly cron, or the feedback batch script.

## 📋 Overview

The project covers two complementary areas:

| Area | Scope |
|:-----|:------|
| 🧠 **Deep Learning** | Multi-class classification (ResNet18: `good`, `broken_large`, `broken_small`, `contamination`) and detection (YOLOv8) on industrial images from MVTec AD, with progressive fine-tuning and bounding-box localization. |
| ⚙️ **MLOps** | Data versioning (DVC), experiment tracking (MLflow), inference API (FastAPI), containerization (Docker), CI/CD/CT (GitHub Actions + kind), and a feedback loop for active learning. |

## 🏗 Architecture

![Architecture](Architecture_animated.svg)

| Stage | Description |
|:------|:------------|
| 📊 **1. Data & Training** | The dataset is versioned with DVC. Training pipelines produce models serialized to ONNX, and MLflow tracks every experiment. |
| 🌐 **2. Serving** | The FastAPI service loads the ONNX models and exposes `/predict` (classifier + YOLO) and `/feedback`. The React frontend supports image upload, webcam capture, and bounding-box visualization. |
| 🔄 **3. MLOps** | User-reported errors are stored in `data/feedback/` and trigger Continuous Training (CT) through GitHub Actions. Docker orchestrates the whole stack. |

## ✨ Features

- 🤖 **Dual-model inference**: ResNet18 defect-type classifier (`good`, `broken_large`, `broken_small`, `contamination`) followed by YOLOv8 (defect localization).
- 🏷 **Defect-type badge**: the API returns `defect_type` and per-class probabilities; the frontend displays a localized badge (FR/EN) and one probability bar per class.
- 🎨 **Modern React interface**: Tailwind-based design, i18n (FR/EN), real-time camera input, feedback dropdown by defect type, and bounding-box overlay.
- 🔁 **Feedback loop**: a "Report an error" flow saves the image and its corrected label to `data/feedback/`.
- 🗂 **Full versioning**: Git (code + ONNX serving artefacts), DVC (data), MLflow (experiments).
- ✅ **Automated tests**: API contract, multi-class data checks, model factory, and non-regression metrics.
- 🛠 **GitHub Actions CI**: installs `libgl1` and skips YOLO tests when the model is not available.
- 🚀 **CD**: builds the API image, deploys 2 replicas to kind, runs `/health` + `/predict` smoke tests, then pushes to GHCR on `main`.

## 📦 Getting Started

### Prerequisites

| Requirement | Version |
|:------------|:--------|
| 🐍 Python | 3.11 |
| 🟢 Node.js | 22+ |
| 🐳 Docker Desktop | Latest |
| ☸️ Minikube | Latest |

### 1. Backend & API

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 2. Data

```bash
# Download MVTec AD and place it in data/raw/
dvc pull  # Fetch the versioned data
```

> The serving ONNX files under `model_registry/` are versioned in Git. The API, Docker build and CI/CD runners therefore start without a `dvc pull`.

### 3. Frontend

```bash
cd frontend
npm install
cd ..
```

## 🚀 Usage

### 🐳 Docker (recommended)

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

| Service | URL |
|:--------|:----|
| 🖥 Frontend | http://localhost:5173 |
| 📖 API (Swagger) | http://localhost:8000/docs |

### 🏋️ Training & Evaluation

```bash
# 1. Multi-class classifier (classes inferred from data/raw/bottle/test)
python pipelines/train_pipeline.py --params configs/params.yaml

# 2. Export to ONNX (fidelity check PyTorch/ONNX < 1e-4)
python scripts/export_model.py

# 3. Evaluate binary + multi-class metrics and confusion matrix
python -m src.models.evaluate

# 4. YOLO detector
python scripts/train_yolo.py
```

### 🧪 Tests

```bash
pytest tests -v
```

## 🧠 Deep Learning Pipeline

| Step | File | Description |
|:----:|:-----|:------------|
| 1 | `src/models/train.py` | Multi-class ResNet18 (class-weighted CrossEntropy), progressive fine-tuning, MLflow tracking, early stopping, and `classes.json`. |
| 2 | `scripts/train_yolo.py` | YOLOv8 training on bounding boxes (15 categories). |
| 3 | `src/models/evaluate.py` | Binary gates (accuracy/precision/recall/F1, mAP, AUROC) + multi-class accuracy/F1 macro, per-class F1 and a true confusion matrix (CSV/PNG in `reports/figures/`). |
| 4 | `scripts/export_model.py` | ONNX export sized from `classes.json`, with fidelity check (max diff < 1e-4). |

## ⚙️ MLOps Pipeline

### 🔄 Automation (CI / CD / CT / Monitoring)

| Workflow | Trigger | Action |
|:---------|:--------|:-------|
| **CI** | Push / Pull Request | Installs `libgl1` and runs `pytest` (YOLO tests skipped if model absent). |
| **CT** | **Push on `data/feedback/**`**, **cron (Monday 03:00 UTC)** + manual | Prepares the mini dataset, retrains, evaluates, runs non-regression tests and commits updated metrics. |
| **CD** | `workflow_run` on CI/CT success (main) + manual | Builds the API image, creates a kind cluster, deploys 2 replicas, runs `/health` + `/predict` smoke tests, then pushes `sha` and `latest` tags to GHCR on `main`. |
| **Monitoring** | Manual (`workflow_dispatch`) | Runs Evidently drift report and uploads it as an artifact. |

#### 🔁 Continuous Training triggers

The CT workflow can be started in three ways:

1. **Push on feedback** — any push touching `data/feedback/**` starts a retrain.
2. **Scheduled run** — every Monday at 03:00 UTC. If `data/feedback/` contains no image, the retrain steps are skipped.
3. **Batch script** — publish only when enough new feedback is available:

```bash
# Dry run: show what would be committed
python scripts/auto_push_feedback.py --dry-run

# Commit and push at least 5 new images (default)
python scripts/auto_push_feedback.py --n 5

# Same through Make
make feedback-push
```

The batch size can also be set with `FEEDBACK_BATCH_SIZE` (default `5`).


### ☸️ Kubernetes (required)

```bash
minikube start
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
minikube service industrial-defect-api
```

Two replicas of the API are deployed, guaranteeing high availability and load balancing. The deployment uses `imagePullPolicy: IfNotPresent`; the CD workflow replaces the image tag with the immutable `<sha>` tag after `kind load docker-image`.

### 📈 Monitoring (Evidently)

The drift monitor compares `data/raw/bottle/train/good` (reference) with `data/feedback` (production):

```bash
python scripts/monitoring_drift.py
```

It writes `reports/drift_report.html` and `reports/drift_summary.json`. The `drift_threshold` (10%) from `configs/alerting.yaml` raises an alert when exceeded. The workflow is available manually through the **Data Drift Monitoring** GitHub Actions workflow.

### 🚨 Alerting

Thresholds are defined in `configs/alerting.yaml`:

| Metric | Threshold | Action |
|:-------|:---------:|:-------|
| Accuracy | 85% | Alert if below |
| F1-score | 80% | Alert if below |
| mAP | 70% | Alert if below |
| Data drift | 10% | Alert if above |

The CI gate blocks the promotion of a model whose metrics fall below these thresholds. These thresholds intentionally stay on the **binary** contract (`good` vs any defect) so that CT and `test_regression.py` remain comparable across releases.

### 📦 Model artefact versioning

| Artefact | Size | Storage |
|:---------|:-----|:--------|
| `model_registry/model.onnx` | ~42.6 MB | **Git** |
| `model_registry/model.onnx.data` | ~42.6 MB | **Git** |
| `model_registry/yolo_best.onnx` | ~12.2 MB | **Git** |
| `model_registry/model.pt` | ~44.8 MB | Local / DVC (ignored by Git) |

All ONNX files are below GitHub's 100 MB per-file limit, so they are versioned directly in Git (no Git LFS required). `.pt` checkpoints remain outside Git. The obsolete `yolo_best.onnx.dvc` pointer was removed because the ONNX is now Git-tracked.

### 📊 Results

Binary metrics are the release gate (current `baseline_metrics.json`). Multi-class metrics are produced by `python -m src.models.evaluate` after retraining and stored in `model_registry/test_metrics.json`.

| Scope | Accuracy | Precision | Recall | F1 | mAP | AUROC |
|:------|:--------:|:---------:|:------:|:--:|:---:|:-----:|
| **Binary** (`good` vs defect) | 85.54% | 100.00% | 80.95% | 89.47% | 99.80% | 99.37% |
| **Multi-class** (4 classes) | *run evaluate* | *run evaluate* | *run evaluate* | *run evaluate* | — | — |

The multi-class confusion matrix is saved to `reports/figures/confusion_matrix.csv` and `reports/figures/confusion_matrix.png`.

<div align="center">

Built with **PyTorch**, **YOLOv8**, **FastAPI**, **DVC**, **MLflow**, **Docker** and **Kubernetes** 🚀

</div>
