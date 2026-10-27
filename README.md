# industrial-defect-detection

Detection de defauts industriels sur **MVTec AD** avec un module Deep Learning
(PyTorch) et un module MLOps (CI/CD/CT, DVC, MLflow, FastAPI, Docker,
Kubernetes/Minikube, monitoring et frontend React).

> Statut : squelette de projet. Les signatures, la configuration, l'API, les
> tests et les manifests sont en place ; les traitements metier (dataset,
> entrainement, metriques, export, drift) sont a implementer.

## Description

Le projet couvre deux axes :

- **Deep Learning** : classification / detection d'anomalies sur les images
  MVTec AD (catégories telles que `bottle`, `metal_nut`, `screw`, ...).
  Baseline : ResNet50, 2 classes (`good` / `defect`), images 224x224.
- **MLOps** : versionnement des donnees avec DVC, tracking des experiences avec
  MLflow, API d'inference FastAPI, conteneurisation Docker, deploiement
  Kubernetes, integration et livraison continues GitHub Actions, re-entrainement
  continu (CT) declenche par de nouvelles donnees, surveillance de la derive.

## Structure

```text
industrial-defect-detection/
├── .github/workflows/        # CI, CT (retrain), CD (deploy)
├── configs/                  # params.yaml, alerting.yaml
├── data/                     # raw/, processed/, new_data/ (DVC, hors Git)
├── docker/                   # Dockerfiles + docker-compose
├── frontend/                 # UI React (Vite)
├── k8s/                      # deployment, service, hpa
├── model_registry/           # baseline_metrics.json + artefacts (hors Git)
├── notebooks/                # EDA, experiences, analyse d'evaluation
├── pipelines/                # train_pipeline.py, prefect_flow.py
├── reports/                  # figures et rapports
├── src/                      # config, data, models, api, monitoring, utils
├── tests/                    # tests unitaires et non-regression
├── dvc.yaml / dvc.lock       # pipeline de donnees reproductible
├── Makefile
└── requirements.txt
```

## Installation

Prerequis : Python 3.11, Git, DVC (installe via `requirements.txt`), Node.js 22+
pour le frontend, Docker et Minikube/Kubectl pour le deploiement.

```bash
python3.11 -m venv .venv
# Windows : py -3.11 -m venv .venv
source .venv/bin/activate            # Windows : .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env 2>/dev/null || true
```

Frontend :

```bash
cd frontend
npm install
cd ..
```

Donnees (MVTec AD n'est pas versionne dans Git) :

```bash
# telecharger MVTec AD depuis https://www.mvtec.com/company/research/datasets/mvtec-ad
# puis placer les categories dans data/raw/
dvc add data/raw
git add data/raw.dvc data/.gitignore .gitignore
git commit -m "data: track MVTec AD with DVC"
```

## Utilisation

```bash
make help            # liste des commandes
make install         # venv + dependances
make train           # pipeline d'entrainement
make test            # pytest
make api             # http://localhost:8000/docs
make frontend        # http://localhost:5173
make docker-build
make docker-up       # API :8000, frontend :5173
make dvc-init
make dvc-pull
make clean
```

Endpoints principaux :

| Methode | Route      | Description                              |
|---------|------------|------------------------------------------|
| GET     | `/health`  | Etat du service et du modele charge      |
| POST    | `/predict` | Upload d'image -> classe + confiance     |
| GET     | `/docs`    | Documentation OpenAPI                    |

Exemple :

```bash
curl -F "file=@data/raw/bottle/test/broken_large/000.png" \
     http://localhost:8000/predict
```

## Phase DL

1. EDA dans `notebooks/01_eda.ipynb` (distribution des classes, tailles,
   desequilibre, exemples de defauts).
2. Entrainement dans `src/models/train.py` : split train/val, early stopping,
   suivi MLflow (parametres, metriques, artefacts).
3. Evaluation dans `src/models/evaluate.py` : accuracy, F1, mAP et matrice de
   confusion ; comparaison au baseline `model_registry/baseline_metrics.json`.
4. Export ONNX / TorchScript via `src/utils/export.py` pour l'inference.

## Phase MLOps

- **CI** (`ci.yml`) : lint/tests sur chaque push et pull request.
- **CT** (`ct-retrain.yml`) : nouveau lot dans `data/new_data/**` -> DVC add,
  re-entrainement, tests de non-regression, mise a jour du baseline si succes.
- **CD** (`cd-deploy.yml`) : apres une CI verte -> build/push des images Docker,
  `kubectl apply` sur Minikube.
- **Monitoring** : `src/monitoring/drift.py` calcule la derive des features et
  compare les metriques aux seuils de `configs/alerting.yaml`.

### Seuils d'alerte

| Metrique   | Seuil |
|------------|-------|
| accuracy   | 0.85  |
| f1         | 0.80  |
| mAP        | 0.70  |
| drift      | 0.10  |

Une metrique inferieure au seuil (ou une derive superieure au seuil) bloque la
promotion du modele et declenche une alerte.

### Kubernetes (Minikube)

```bash
minikube start
eval $(minikube docker-env)          # PowerShell : & minikube -p minikube docker-env | Invoke-Expression
docker build -f docker/Dockerfile.api -t industrial-defect-detection/api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
minikube service industrial-defect-api
```

## Livrables

- [ ] Rapport d'EDA (`reports/`)
- [ ] Modele baseline + metriques (`model_registry/baseline_metrics.json`)
- [ ] API FastAPI conteneurisee
- [ ] Interface React de demonstration
- [ ] Pipeline CI/CD/CT operationnel
- [ ] Deploiement Kubernetes (2 replicas, HPA 2-5)
- [ ] Rapport de derive et procedure d'alerte
