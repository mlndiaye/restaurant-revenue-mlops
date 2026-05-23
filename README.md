# Restaurant Revenue Prediction — MLOps Project

Prédiction des ventes annuelles de restaurants pour TFI (Burger King, Sbarro, Popeyes, Arby's)
à partir de données démographiques, immobilières et commerciales.

## Stack technique

- **Data versioning** : DVC
- **Experiment tracking** : MLflow
- **Modèles** : scikit-learn, XGBoost, LightGBM
- **Interprétabilité** : SHAP
- **Serving** : FastAPI
- **Containerisation** : Docker
- **CI/CD** : GitHub Actions

## Structure du projet

```
├── notebooks/          # EDA et modélisation
├── data/               # Données (gérées par DVC, non commitées)
│   ├── raw/
│   └── processed/
├── src/                # Scripts Python réutilisables
├── api/                # FastAPI serving
├── tests/              # Tests unitaires
└── .github/workflows/  # CI/CD
```

## Prérequis

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — gestionnaire de paquets

```bash
# Installer uv si ce n'est pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
# Cloner le projet
git clone <url-du-repo>
cd restaurant-revenue-mlops

# Créer l'environnement virtuel et installer les dépendances
uv sync --dev

# Activer l'environnement
source .venv/bin/activate
```

## Dataset

Télécharger le dataset depuis Kaggle :
[Restaurant Revenue Prediction](https://www.kaggle.com/competitions/restaurant-revenue-prediction/data)

Placer les fichiers dans `data/raw/` :
```
data/raw/train.csv
data/raw/test.csv
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/resto_revenue_01_analyse.ipynb` | Analyse exploratoire des données |
| `notebooks/resto_revenue_02_modelisation.ipynb` | Tests de modèles et sélection du modèle final |

Lancer Jupyter :
```bash
jupyter notebook
```

## Métriques

Métrique principale : **RMSE** (Root Mean Squared Error)
