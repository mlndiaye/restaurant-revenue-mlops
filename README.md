# Restaurant Revenue Prediction — MLOps Project

> Prédiction des ventes annuelles de restaurants pour **TFI** (Burger King, Sbarro, Popeyes, Arby's)
> à partir de données démographiques, immobilières et commerciales.

**Contexte** : TFI ouvre ~1 200 restaurants à travers l'Europe et l'Asie. Un mauvais emplacement entraîne une fermeture dans les 18 mois. L'objectif est de prédire le revenu annuel d'un site avant son ouverture pour guider les décisions d'investissement.

---

## Stack technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| Versioning données | DVC | Pipeline reproductible, données non committées |
| Experiment tracking | MLflow | Paramètres, métriques, modèles archivés |
| Modélisation | scikit-learn, XGBoost, LightGBM | 5 modèles comparés via nested CV |
| Optimisation | Optuna | Recherche bayésienne des hyperparamètres |
| Serving | FastAPI | API REST avec validation Pydantic |
| Containerisation | Docker | Runtime portable |
| CI/CD | GitHub Actions | Tests automatiques sur chaque push |
| Monitoring | Streamlit + scipy | Dashboard de détection de drift |

---

## Résultats

Protocole : **Nested Cross-Validation** (5 folds externes × 3 folds internes) + Optuna TPE.

| Modèle | RMSE OOF | R² OOF |
|--------|----------|--------|
| **RandomForest** ← modèle retenu | **2 508 158 TRY** | **0.045** |
| XGBoost | 2 515 764 TRY | 0.039 |
| LightGBM | 2 552 830 TRY | 0.011 |
| Baseline (moyenne) | 2 571 302 TRY | -0.004 |
| Ridge | 2 586 861 TRY | -0.016 |
| Lasso | 2 616 176 TRY | -0.039 |

> Le R² faible (~0.045) est attendu : corrélation maximale feature↔revenu r=0.19, 137 observations, features obfusquées.

---

## Structure du projet

```
├── notebooks/
│   ├── 01_eda.ipynb              # Analyse exploratoire
│   └── 02_modelisation.ipynb     # Nested CV + Optuna + sélection du modèle
├── src/restaurant_revenue/
│   ├── data/load.py              # Chargement des données
│   ├── features/preprocessing.py # Pipeline de preprocessing (anti data-leakage)
│   ├── models/
│   │   ├── experimentation.py    # Nested CV + Optuna (réutilisable)
│   │   ├── train.py              # Entraînement du modèle final
│   │   └── evaluate.py           # Évaluation + métriques DVC
│   └── utils/drift.py            # Détection de drift
├── monitoring/dashboard.py       # Dashboard Streamlit
├── api/main.py                   # FastAPI — /health + /predict
├── tests/test_api.py             # Tests unitaires
├── dvc.yaml                      # Pipeline DVC (4 étapes)
├── Dockerfile
└── .github/workflows/ci.yml      # CI/CD
```

---

## Installation

**Prérequis** : Python 3.12, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/mlndiaye/restaurant-revenue-mlops.git
cd restaurant-revenue-mlops
uv sync
```

### LightGBM et XGBoost — OpenMP

Ces deux librairies s'appuient sur OpenMP pour le parallélisme.

| OS | Action requise |
|----|----------------|
| **macOS** | `brew install libomp` — OpenMP absent du compilateur Apple par défaut |
| **Linux** | Rien — `libgomp1` inclus avec GCC (géré dans le Dockerfile) |
| **Windows** | Rien — les wheels embarquent directement les DLLs OpenMP |

Sans `libomp` sur macOS, l'import de LightGBM lève une erreur au démarrage.

---

## Dataset

Télécharger depuis Kaggle :
[Restaurant Revenue Prediction](https://www.kaggle.com/competitions/restaurant-revenue-prediction/data)

Placer dans `data/raw/` :
```
data/raw/train.csv
data/raw/test.csv
```

---

## Notebooks

Les notebooks utilisent le kernel du virtualenv `.venv` créé par `uv sync`.

### VS Code (recommandé)

1. Ouvrir le dossier dans VS Code
2. Installer l'extension **Jupyter** si ce n'est pas déjà fait
3. Ouvrir `notebooks/01_eda.ipynb` ou `notebooks/02_modelisation.ipynb`
4. En haut à droite, cliquer sur **"Select Kernel"** → **"Python Environments"** → choisir `.venv`
5. Exécuter les cellules normalement

### JupyterLab (terminal)

```bash
uv run jupyter lab
```

Ouvre JupyterLab dans le navigateur. Le kernel `.venv` est automatiquement disponible.

---

## Reproduire le pipeline complet

```bash
# Prétraitement + entraînement + évaluation en une commande
PYTHONPATH=src uv run dvc repro
```

Étapes du pipeline :
```
load_data → preprocess → train → evaluate
```

---

## Visualiser les expériences MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
# → http://localhost:5000
```

---

## API de prédiction

```bash
PYTHONPATH=src uv run uvicorn api.main:app --reload
# → http://localhost:8000/docs  (Swagger UI)
```

Exemple de requête :
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Open Date": "01/01/2010", "City": "Istanbul",
    "City Group": "Big Cities", "Type": "FC",
    "P1":5,"P2":3,"P3":2,"P4":1,"P5":0,"P6":4,"P7":2,"P8":1,"P9":0,"P10":3,
    "P11":1,"P12":0,"P13":2,"P14":0,"P15":0,"P16":0,"P17":0,"P18":0,"P19":1,"P20":2,
    "P21":0,"P22":1,"P23":0,"P24":0,"P25":0,"P26":0,"P27":0,"P28":3,"P29":1,"P30":0,
    "P31":0,"P32":0,"P33":0,"P34":0,"P35":0,"P36":0,"P37":0
  }'
# → {"prediction": 4690480.8}
```

---

## Dashboard de monitoring

```bash
PYTHONPATH=src uv run streamlit run monitoring/dashboard.py
# → http://localhost:8501
```

Compare la distribution des données de production aux données d'entraînement.
Détecte le drift via KS-test (numérique) et Chi² (catégoriel). Alerte si >10% des features dérivent.

---

## Docker

```bash
docker build -t restaurant-revenue .
docker run -p 8000:8000 restaurant-revenue
```

---

## Tests

```bash
PYTHONPATH=src uv run pytest tests/ -v
```
