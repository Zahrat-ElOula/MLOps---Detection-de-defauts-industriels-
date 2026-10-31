<div align="center">

# 🏭 Industrial Defect Detection

**Détection de défauts sur le dataset MVTec AD, combinant un module Deep Learning (PyTorch/YOLO) et une stack MLOps complète.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF?logo=ultralytics&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-inference%20API-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?logo=react&logoColor=black)
![DVC](https://img.shields.io/badge/DVC-data%20versioning-945DD6?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-containers-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=github-actions&logoColor=white)

[Architecture](#-architecture) · [Fonctionnalités](#-features) · [Installation](#-installation) · [Usage](#-usage)

</div>

---

> [!NOTE]
> **Statut : Projet terminé.** Le système est fonctionnel de bout en bout, avec une architecture à deux étages (classification + détection), une API conteneurisée, une interface web moderne et une boucle de rétroaction pour le Continuous Training.

## 📋 Overview

Le projet couvre deux volets complémentaires :

| Volet | Périmètre |
|-------|-----------|
| 🧠 **Deep Learning** | Classification (ResNet18) et détection (YOLOv8) sur les images industrielles (MVTec AD). Gestion du Fine-Tuning progressif et des bounding boxes. |
| ⚙️ **MLOps** | Data versioning (DVC), tracking (MLflow), API FastAPI, conteneurisation (Docker), CI/CD (GitHub Actions), et boucle de Feedback (Active Learning). |

## 🏗 Architecture

![Architecture](https://raw.githubusercontent.com/Zahrat-ElOula/MLOps---Detection-de-defauts-industriels-/main/docs/architecture.png)
*(Note : Si l'image ne s'affiche pas, veuillez vous référer à la capture ci-dessus ou au diagramme Draw.io inclus dans le dossier `docs/`)*

1. **Data & Training** : Le dataset est versionné par DVC. Les pipelines d'entraînement génèrent des modèles sérialisés en ONNX. MLflow trace chaque expérience.
2. **Serving** : L'API FastAPI charge les modèles ONNX et expose `/predict` (classifieur + YOLO) et `/feedback`. Le Frontend React permet l'upload, la webcam et l'affichage des bounding boxes.
3. **MLOps** : Les erreurs signalées par les utilisateurs sont stockées dans `data/feedback/` pour déclencher le Continuous Training (CT) via GitHub Actions. Docker orchestre le tout.

## 🚀 Features

- **Double modèle** : ResNet18 (bonne/mauvaise pièce) + YOLOv8 (localisation du défaut).
- **Interface React** : Design moderne (Tailwind), i18n (FR/EN), caméra temps réel, overlay des bounding boxes.
- **Boucle MLOps** : Bouton "Signaler une erreur" qui sauvegarde l'image et le label corrigé dans `data/feedback/`.
- **Versioning** : Git (code), DVC (données), MLflow (expériences).
- **17 Tests** : Validation de l'API, des données et des métriques de non-régression.
- **CI GitHub Actions** : Workflow qui installe `libgl1` et skippe les tests YOLO si le modèle n'est pas présent.

## 📦 Installation

### Prérequis
- Python 3.11
- Node.js 22+
- Docker Desktop

### 1. Backend & API
```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
