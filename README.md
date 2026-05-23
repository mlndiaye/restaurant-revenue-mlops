# Restaurant Revenue Prediction — MLOps Project

Projet MLOps de prédiction des ventes annuelles de restaurants pour TFI (Burger King, Sbarro, Popeyes, Arby's).

## Objectif

Prédire les ventes annuelles de nouveaux restaurants à partir de données démographiques, immobilières et commerciales (dataset Kaggle).

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
├── data/               # Données (gérées par DVC)
│   ├── raw/
│   └── processed/
├── src/                # Scripts Python réutilisables
├── api/                # FastAPI serving
├── tests/              # Tests unitaires
└── .github/workflows/  # CI/CD
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `resto_revenue_01_analyse.ipynb` | Analyse exploratoire des données |
| `resto_revenue_02_modelisation.ipynb` | Tests de modèles et sélection du modèle final |

## Lancement

### Entraînement

```bash
pip install -r requirements.txt
dvc repro
```

### API de prédiction

```bash
uvicorn api.main:app --reload
# ou
docker build -t restaurant-revenue-api .
docker run -p 8000:8000 restaurant-revenue-api
```

### MLflow UI

```bash
mlflow ui
# Accessible sur http://localhost:5000
```

## Métriques

Métrique principale : **RMSE** (Root Mean Squared Error)
