# Comprendre le projet de bout en bout

> Ce fichier est ignoré par git (`.gitignore`). Notes personnelles pour la soutenance.

---

## Table des matières

1. [Le problème business](#1-le-problème-business)
2. [Les données — ce qu'on a découvert](#2-les-données--ce-quon-a-découvert)
3. [Le preprocessing — préparer les données sans tricher](#3-le-preprocessing--préparer-les-données-sans-tricher)
4. [La stratégie de modélisation](#4-la-stratégie-de-modélisation)
5. [Les métriques — comment lire les résultats](#5-les-métriques--comment-lire-les-résultats)
6. [Les résultats et pourquoi ils sont normaux](#6-les-résultats-et-pourquoi-ils-sont-normaux)
7. [La couche MLOps — chaque outil et son rôle](#7-la-couche-mlops--chaque-outil-et-son-rôle)
8. [Le flux complet du projet](#8-le-flux-complet-du-projet)
9. [Questions jury probables et réponses](#9-questions-jury-probables-et-réponses)

---

## 1. Le problème business

**TFI** (Türkiye Fast Food International) gère plus de 1 200 restaurants en Europe et en Asie : Burger King, Popeyes, Arby's, Sbarro.

Chaque nouvelle ouverture représente des millions d'investissement. Actuellement, la décision se prend à l'instinct — l'équipe de développement visite un emplacement, juge subjectivement, et décide. Résultat : un restaurant sur plusieurs ferme dans les **18 premiers mois** parce que le site n'était pas bon.

**L'objectif du projet** : construire un modèle prédictif qui estime le revenu annuel d'un restaurant **avant son ouverture**, à partir de données disponibles sur le site (démographie du quartier, données immobilières, données commerciales). Si le modèle prédit un revenu insuffisant pour atteindre le seuil de rentabilité, on ne construit pas.

---

## 2. Les données — ce qu'on a découvert

### Structure du dataset

- **137 restaurants** d'entraînement avec leur revenu annuel réel → c'est ce qu'on utilise pour apprendre
- **100 000 restaurants** à prédire → on ne connaît pas leur revenu
- **40 features** par restaurant : données démographiques, immobilières, commerciales

> **Vocabulary** : une **feature** (ou variable explicative) est une information qu'on donne au modèle pour l'aider à prédire. Ici : l'âge du restaurant, la ville, le type, les mesures P1 à P37.

**Problème majeur : 137 c'est très peu.** Pour donner un ordre d'idée, les modèles de vision par ordinateur s'entraînent sur des millions d'images. Ici on a 137 exemples pour apprendre à prédire un phénomène économique complexe. Toutes les décisions techniques qui suivent découlent de cette contrainte.

### Les features obfusquées

Les features P1 à P37 sont anonymisées — TFI ne révèle pas ce qu'elles représentent. P14 pourrait être "nombre de places de parking", P22 "densité de population dans un rayon de 500m", etc. On ne sait pas. On travaille avec des signaux aveugles.

### La distribution du revenu

| Statistique | Valeur |
|-------------|--------|
| Minimum | 1 149 870 TRY |
| Médiane | 3 939 804 TRY |
| Moyenne | 4 453 533 TRY |
| Maximum | 19 696 939 TRY |
| Skewness | **2.793** |

> **Vocabulary** : la **skewness** (asymétrie) mesure à quel point une distribution s'écarte d'une forme symétrique en cloche. Une skewness de 0 = symétrique. Au-delà de 1 = fortement asymétrique.

La moyenne (4.45M) dépasse la médiane (3.94M) parce que quelques restaurants d'Istanbul à 15-20M TRY tirent la moyenne vers le haut — comme en démographie où la richesse de quelques milliardaires fait monter la moyenne nationale très au-dessus du revenu médian.

**Solution : travailler en log(revenue)**

La transformation logarithmique compresse les valeurs extrêmes et ramène la distribution vers une forme gaussienne (en cloche) : la skewness passe de **2.79 → 0.31**.

Concrètement : un modèle entraîné sur log(revenue) traite l'écart entre 1M et 2M (×2) de la même façon que l'écart entre 10M et 20M (×2). Sans log, le modèle serait obsédé par les restaurants d'Istanbul et négligerait les autres.

### Les features sparses

17 des 37 features P ont plus de **64% de zéros**. Ce sont probablement des comptages qui ne s'appliquent qu'à certains types de sites (nombre d'accès handicapés, de places de livraison, etc.). Pour un restaurant ordinaire, ces features valent 0.

> **Vocabulary** : une feature **sparse** (éparse) est une feature dont la grande majorité des valeurs est nulle ou absente.

On a créé des features binaires `Px_active` (1 si la valeur > 0, sinon 0) pour aider les modèles linéaires à exploiter ce signal.

### La corrélation des features avec le revenu

La corrélation de Pearson mesure la force d'une relation linéaire entre deux variables. Elle va de -1 (relation inverse parfaite) à +1 (relation directe parfaite), 0 signifiant aucune relation linéaire.

| Feature | Corrélation avec revenue |
|---------|--------------------------|
| `days_since_open` (créée par nous) | **r = 0.326** |
| P2 (meilleure feature originale) | r = 0.192 |
| P28 | r = 0.156 |
| P6 | r = 0.139 |
| ... | ... |

**Insight clé** : la feature que nous avons fabriquée (`days_since_open`) est **la plus corrélée avec le revenu de tout le dataset** — plus que n'importe laquelle des 37 features originales de TFI. Les restaurants ouverts depuis longtemps tendent à avoir un revenu plus élevé (maturité de la clientèle, notoriété locale accumulée).

La corrélation maximale de **r = 0.192** pour les features P est très faible. Cette valeur aura une importance cruciale pour interpréter les résultats du modèle.

---

## 3. Le preprocessing — préparer les données sans tricher

### Deux types de transformations

Le preprocessing transforme les données brutes en données utilisables par le modèle. Il y a deux catégories fondamentalement différentes.

**Transformations stateless** (sans mémoire) — ne dépendent que de la ligne elle-même :
- Calculer `days_since_open` = différence de dates
- Créer les flags `Px_active`
- Supprimer les colonnes inutiles (`Id`, `Open Date` brut)

Ces transformations peuvent être appliquées sur tout le dataset sans risque, car elles ne "apprennent" rien des données.

**Transformations stateful** (avec mémoire) — doivent apprendre sur les données :
- **StandardScaler** : calcule la moyenne µ et l'écart-type σ de chaque feature numérique, puis soustrait µ et divise par σ pour tout ramener à la même échelle. Exemple : si P1 varie entre 0 et 5 et P30 entre 0 et 500, le modèle linéaire penserait que P30 est 100× plus importante. Le scaling corrige ça.
- **OneHotEncoder** : transforme une variable catégorielle (ex: "Big Cities", "Other") en colonnes binaires (0 ou 1). C'est nécessaire car les modèles mathématiques ne peuvent pas travailler avec du texte.

### Pourquoi les transformations stateful doivent rester dans le Pipeline

C'est le point le plus important de tout le preprocessing.

**Le problème de la fuite de données (data leakage)**

Imagine que tu veux évaluer un modèle médical pour détecter une maladie rare. Tu as 100 patients. Tu calcules d'abord la moyenne et l'écart-type de tous les marqueurs biologiques sur les 100 patients, puis tu divises : 80 pour entraîner, 20 pour tester.

Problème : le modèle a déjà "vu" les 20 patients de test pendant le calcul des statistiques. Il est légèrement adapté à eux. Sa performance mesurée sera trop optimiste — elle ne reflète pas sa vraie capacité à traiter de nouveaux patients.

C'est la **fuite de données** : de l'information provenant du set de validation "fuit" dans l'entraînement.

**La solution : le Pipeline sklearn**

Le Pipeline garantit que StandardScaler et OneHotEncoder :
- Apprennent leurs statistiques (µ, σ, catégories) **uniquement sur les données d'entraînement**
- Appliquent ces statistiques calculées pour transformer les données de validation, **sans les regarder**

Métaphore : c'est comme étalonner une balance uniquement avec des poids certifiés (données d'entraînement), puis peser des inconnus avec cette balance étalonnée — et non pas étalonner la balance en incluant les inconnus dans le processus.

---

## 4. La stratégie de modélisation

### Pourquoi la cross-validation classique ne suffit pas

> **Vocabulary** : la **cross-validation** (validation croisée) est une technique d'évaluation qui découpe les données en K parties (folds), entraîne sur K-1 parties et teste sur la K-ième, répète K fois, et moyenne les résultats. Elle donne une meilleure estimation de la vraie performance qu'un simple split train/test.

Avec seulement 137 lignes, chaque observation compte énormément. Un simple split 80/20 donnerait un ensemble de test de 27 lignes — si ces 27 lignes contiennent 3 restaurants d'Istanbul à 15M TRY par hasard, la mesure de performance sera faussée.

La cross-validation à 5 folds est déjà mieux : chaque restaurant est dans l'ensemble de test exactement une fois.

**Mais il y a un deuxième problème** : on veut aussi optimiser les hyperparamètres.

> **Vocabulary** : les **hyperparamètres** sont les réglages du modèle qu'on choisit avant l'entraînement (vs les paramètres appris pendant l'entraînement). Exemple pour RandomForest : le nombre d'arbres (`n_estimators`), la profondeur maximale de chaque arbre (`max_depth`). Ces valeurs ne sont pas apprises par le modèle — on les choisit.

Si on utilise les mêmes 5 folds pour optimiser les hyperparamètres ET pour évaluer le modèle, on tombe dans le même piège que la data leakage : l'évaluation est biaisée vers la configuration qui fonctionne bien sur ces folds précis.

### La Nested Cross-Validation

La solution est une boucle à deux niveaux.

**Boucle externe (5 folds)** — pour l'évaluation honnête :
- Fold 1 validation : restaurants 1-27 → ne sont JAMAIS vus pendant l'optimisation
- Fold 2 validation : restaurants 28-54 → idem
- ...

**Boucle interne (3 folds)** — pour l'optimisation des hyperparamètres :
- Sur les 110 restaurants restants après avoir mis de côté le fold externe
- Optuna cherche les meilleurs hyperparamètres en testant différentes configurations
- Ces 3 folds internes ne voient JAMAIS les 27 restaurants du fold externe

**Résultat** : chaque restaurant est prédit par un modèle qui n'a jamais eu accès à lui, ni directement pendant l'entraînement, ni indirectement via l'optimisation. Ces 137 prédictions s'appellent les **OOF predictions** (Out-Of-Fold).

Analogie scientifique : c'est l'équivalent d'un essai clinique en double aveugle. Le protocole d'optimisation du traitement (boucle interne) et l'évaluation finale de son efficacité (boucle externe) sont strictement séparés. Contaminer l'un par l'autre invaliderait les résultats.

### Optuna — l'optimisation bayésienne des hyperparamètres

Pour RandomForest par exemple, les hyperparamètres à optimiser incluent :
- `n_estimators` : entre 100 et 500 arbres
- `max_depth` : entre 3 et 15 niveaux
- `max_features` : 'sqrt', 'log2', ou une valeur entre 0.1 et 1.0
- `min_samples_leaf` : entre 1 et 10

Tester toutes les combinaisons (grille exhaustive) : 5 × 13 × 4 × 10 = **2 600 configurations**. Multiplié par 3 folds internes × 5 folds externes = **39 000 entraînements**. Trop long.

**L'approche naïve vs Optuna** :

La grille exhaustive teste tout aveuglément.

Optuna utilise un algorithme appelé **TPE (Tree-structured Parzen Estimator)**. Il modélise la relation entre les hyperparamètres et la performance de la même façon qu'un chercheur affine son hypothèse au fil de ses expériences : après avoir vu que `max_depth=8` donne de meilleurs résultats que `max_depth=3` et `max_depth=15`, il va concentrer ses essais suivants autour de 7-10 sans tester inutilement des valeurs extrêmes.

**Le MedianPruner** arrête les essais clairement mauvais après quelques folds internes — si un essai donne déjà un RMSE 50% plus élevé que la médiane des essais précédents après 1 fold, inutile de continuer. C'est l'équivalent d'un arrêt précoce dans un essai clinique quand un groupe de traitement montre clairement de moins bons résultats.

### Le TransformedTargetRegressor

Le modèle n'apprend pas directement à prédire `revenue` — il apprend à prédire `log(revenue)`. Le `TransformedTargetRegressor` de sklearn encapsule cette logique :
- À l'entraînement : transforme automatiquement `y → log(y)` avant de passer au modèle
- À la prédiction : applique automatiquement `exp(prédiction)` pour revenir en TRY

Cela garantit que la transformation est cohérente à l'intérieur de la cross-validation — jamais oubliée, jamais appliquée au mauvais moment.

---

## 5. Les métriques — comment lire les résultats

### RMSE — Root Mean Squared Error

**Définition** : la racine carrée de la moyenne des erreurs au carré.

Intuitivement : c'est l'erreur typique du modèle, dans la même unité que la variable cible. Si le RMSE est 2 508 158 TRY, cela signifie que l'erreur typique du modèle est d'environ 2.5 millions de TRY par restaurant.

**Pourquoi mettre les erreurs au carré ?** Pour pénaliser fortement les grandes erreurs. Une erreur de 4M compte 4× plus qu'une erreur de 2M (et non 2×). Cela correspond à la réalité business : se tromper de 5M sur un restaurant est catastrophique, pas juste 2.5× plus grave que de se tromper de 2M.

**Pourquoi RMSE et pas l'erreur absolue moyenne (MAE) ?** Le MAE est plus robuste aux outliers (il traite chaque erreur proportionnellement). Le RMSE est plus sensible aux grosses erreurs, ce qui est pertinent ici : TFI veut éviter les grandes erreurs (ouvrir un restaurant voué à l'échec), pas juste minimiser l'erreur moyenne.

### MAE — Mean Absolute Error

**Définition** : la moyenne des valeurs absolues des erreurs.

Si le MAE est 1 543 509 TRY, cela signifie que *en moyenne*, le modèle se trompe de 1.54 million de TRY. Moins pénalisé par les outliers que le RMSE.

### R² — Coefficient de détermination

**Définition** : la proportion de la variance de la variable cible expliquée par le modèle.

- R² = 1.0 : prédictions parfaites
- R² = 0.0 : le modèle fait aussi bien que prédire la moyenne à chaque fois
- R² < 0 : le modèle fait moins bien que prédire la moyenne (pire que le modèle le plus naïf)

**Exemple concret** : si les restaurants ont des revenus qui varient de 1M à 20M TRY, et que ton modèle explique 4.5% de cette variation (R²=0.045), il capte un signal réel mais très faible. La grande majorité de la variance reste inexpliquée.

**Relation avec la corrélation** : pour un modèle linéaire, R² = r². Si la meilleure corrélation disponible est r = 0.192 (P2), alors le plafond théorique pour un modèle linéaire est R² = 0.192² = **0.037**. Et en effet, Ridge et Lasso (modèles linéaires) ont des R² négatifs ou proches de zéro.

---

## 6. Les résultats et pourquoi ils sont normaux

### Tableau comparatif

| Modèle | RMSE OOF | MAE OOF | R² OOF |
|--------|----------|---------|--------|
| **RandomForest** ← retenu | **2 508 158 TRY** | 1 543 509 TRY | **0.045** |
| XGBoost | 2 515 764 TRY | — | 0.039 |
| LightGBM | 2 552 830 TRY | — | 0.011 |
| Baseline (prédire la moyenne) | 2 571 302 TRY | 1 681 468 TRY | -0.004 |
| Ridge | 2 586 861 TRY | — | -0.016 |
| Lasso | 2 616 176 TRY | — | -0.039 |

### Pourquoi R² = 0.045 n'est pas un échec

C'est la question qui reviendra à coup sûr en soutenance. Voici le raisonnement à tenir.

**Étape 1 : regarder les données**

La corrélation maximale entre une feature et le revenu est r = 0.326 (pour `days_since_open` qu'on a fabriqué) et r = 0.192 pour les features originales.

**Étape 2 : le plafond théorique**

Pour un modèle linéaire parfait sur une seule feature, R² = r². Avec r = 0.192, R² max ≈ 0.037. Les modèles linéaires (Ridge, Lasso) avec toutes les features atteignent effectivement un R² négatif — ce qui signifie que la combinaison linéaire de toutes ces features faiblement corrélées fait **moins bien** que prédire la moyenne.

**Étape 3 : ce que RandomForest fait de plus**

RandomForest capture des relations **non-linéaires** et des **interactions entre features** que la corrélation de Pearson ne mesure pas. C'est pour ça qu'il dépasse le plafond théorique linéaire (0.045 > 0.037). Mais il reste limité par la qualité du signal disponible.

**Étape 4 : le contexte concurrentiel**

Cette compétition Kaggle originale (2015) a réuni des milliers de Data Scientists. Les meilleures solutions publiées atteignaient elles aussi des R² faibles sur ce dataset. Notre R² de 0.045 est cohérent avec l'état de l'art sur ces données spécifiques.

**Conclusion à retenir** : un R² faible sur des données faiblement corrélées n'est pas un modèle raté — c'est un modèle honnête qui ne sur-apprend pas.

### Hyperparamètres du modèle final (RandomForest)

| Paramètre | Valeur | Signification |
|-----------|--------|---------------|
| `n_estimators` | 400 | 400 arbres de décision dans la forêt |
| `max_depth` | 8 | Chaque arbre peut avoir au maximum 8 niveaux |
| `max_features` | 'log2' | Chaque split ne considère que log2(40) ≈ 5 features aléatoires |
| `min_samples_leaf` | 2 | Chaque feuille doit contenir au moins 2 observations |
| `add_sparse_flags` | False | Les flags binaires Px_active n'améliorent pas la performance |

**Pourquoi `max_features='log2'` ?** RandomForest fonctionne en construisant chaque arbre sur un sous-ensemble aléatoire de features. Si tous les arbres utilisaient toutes les features, ils seraient très corrélés entre eux et la moyenne n'apporterait pas grand-chose. En forçant chaque arbre à ne regarder que 5 features parmi 40, on garantit la diversité entre les arbres — et la moyenne de décisions diverses est plus robuste qu'une décision unanime.

**Pourquoi `add_sparse_flags=False` ?** Optuna a testé True et False sur chaque fold. Les résultats étaient incohérents (True gagnait sur certains folds, False sur d'autres) — signal trop bruité sur 137 lignes pour être fiable. Le modèle final a choisi False.

---

## 7. La couche MLOps — chaque outil et son rôle

> **Vocabulary** : **MLOps** (Machine Learning Operations) désigne l'ensemble des pratiques et outils qui permettent de déployer, maintenir et monitorer des modèles de ML en production de façon fiable et reproductible. C'est l'équivalent du DevOps pour le code classique, appliqué aux modèles.

### DVC — Data Version Control

**Problème résolu** : comment s'assurer que n'importe qui peut reproduire exactement les mêmes résultats, dans le bon ordre, sans avoir à lire tout le code pour comprendre quoi lancer ?

**Ce que DVC fait** : il modélise le pipeline comme un graphe de dépendances entre fichiers. `dvc.yaml` déclare 4 étapes :

```
load_data → preprocess → train → evaluate
```

Chaque étape déclare ses **entrées** (fichiers dont elle dépend) et ses **sorties** (fichiers qu'elle produit). Si tu modifies `train.py`, DVC sait que seules les étapes `train` et `evaluate` doivent être relancées — pas `preprocess`. `dvc repro` ne refait que le travail nécessaire.

`dvc.lock` joue le rôle d'un historique cryptographique : il enregistre les hash des fichiers de données à chaque exécution. En committant ce fichier, quelqu'un qui clone le repo et exécute `dvc repro` peut recréer exactement le même état.

### MLflow — Experiment Tracking

**Problème résolu** : comment comparer rigoureusement 5 modèles avec des dizaines de configurations différentes, et retrouver dans 3 semaines pourquoi on a fait tel choix ?

**Ce que MLflow fait** : à chaque `run`, il archive automatiquement :
- les **paramètres** (hyperparamètres, configuration)
- les **métriques** (RMSE, MAE, R²)
- le **modèle sérialisé** (rechargeable pour faire des prédictions)
- des **artefacts** (fichiers JSON avec les détails par fold)

L'UI (`mlflow ui`) permet de trier les runs par métrique, comparer visuellement, et inspecter n'importe quelle configuration passée.

Sans MLflow, chaque expérience produirait des chiffres dans un terminal que personne ne retrouverait.

**L'idempotence** : on a ajouté un mécanisme pour que relancer le notebook ne crée pas de doublons. Avant chaque run, le code cherche et supprime les anciens runs avec le même nom. Résultat : 10 runs propres dans MLflow, pas 15 ou 20.

### FastAPI — Serving

**Problème résolu** : comment rendre le modèle utilisable par des systèmes tiers (application mobile TFI, dashboard web interne, etc.) sans qu'ils aient à installer Python ?

**Ce que FastAPI fait** : il expose le modèle via une API REST (protocole HTTP standard). Deux endpoints :

- `GET /health` → vérifie que le modèle est chargé et l'API est opérationnelle. Utilisé par les systèmes de monitoring pour savoir si le service est en vie.
- `POST /predict` → reçoit les features d'un restaurant au format JSON, retourne le revenu prédit.

**La validation Pydantic** : avant d'appeler le modèle, FastAPI vérifie automatiquement que tous les champs requis sont présents et ont le bon type. Si une requête malformée arrive (P1 absent, City Group invalide), l'API retourne une erreur 422 claire plutôt que de crasher silencieusement.

**Le Swagger automatique** : FastAPI génère `/docs` — une interface web interactive pour tester l'API directement dans le navigateur — sans écrire une ligne de documentation.

**Le chargement au démarrage** : le modèle (`.pkl`) est chargé une seule fois quand le serveur démarre (mécanisme `lifespan`), pas à chaque requête. Si le modèle pèse 50MB, le recharger à chaque requête prendrait des secondes. En le gardant en mémoire, chaque prédiction prend quelques millisecondes.

### Docker

**Problème résolu** : "ça marche sur ma machine" — comment garantir que l'API tourne identiquement partout ?

**Ce que Docker fait** : il crée une **image** — une capsule auto-suffisante qui contient exactement Python 3.12, toutes les dépendances (versions pinnées), le code, et le modèle. Cette image peut être lancée sur n'importe quel serveur Linux, Mac ou Windows sans installation préalable.

Le `Dockerfile` installe aussi `libgomp1` (OpenMP sur Linux) — nécessaire pour LightGBM et XGBoost, qui parallélisent leurs calculs via OpenMP.

### GitHub Actions — CI/CD

> **Vocabulary** : **CI** (Continuous Integration) = vérification automatique que le code fonctionne à chaque changement. **CD** (Continuous Delivery) = déploiement automatique après vérification.

**Problème résolu** : comment savoir immédiatement si une modification casse quelque chose ?

À chaque `git push` sur `main`, GitHub exécute automatiquement :
1. Installation des dépendances (`uv sync`)
2. Génération des données prétraitées (`preprocessing.py`)
3. Entraînement du modèle (`train.py` → génère `best_model.pkl`)
4. Tests unitaires (`pytest tests/`)

Si n'importe quelle étape échoue, le push est marqué en rouge. L'équipe sait immédiatement qu'il y a un problème — avant que ça arrive en production.

### Streamlit — Dashboard de monitoring

**Problème résolu** : comment détecter que les données de production ont changé et que le modèle risque de devenir peu fiable ?

Un modèle est entraîné sur une distribution de données précise. Si les nouvelles données entrant dans l'API ont une distribution différente (nouveau type de quartier, nouvelle échelle de valeurs, nouvelles villes), le modèle applique des règles apprises sur des données qui ne ressemblent plus à ce qu'il reçoit.

**Ce que le dashboard fait** :

Pour chaque feature, il compare la distribution des données de référence (train) aux données de production via :
- **KS-test (Kolmogorov-Smirnov)** pour les features numériques : teste si deux échantillons pourraient venir de la même distribution. Exemple d'usage similaire en physique : tester si deux ensembles de mesures proviennent du même instrument.
- **Chi² (Chi-carré)** pour les features catégorielles : teste si les fréquences observées s'écartent des fréquences attendues.

Si la p-value < 0.05 pour une feature → cette feature a significativement drifté. Si plus de 10% des features driftent → alerte.

> **Vocabulary** : la **p-value** est la probabilité d'obtenir des données aussi extrêmes si l'hypothèse nulle était vraie. Ici, l'hypothèse nulle est "les deux distributions sont identiques". p < 0.05 signifie : il y a moins de 5% de chances que cette différence soit due au hasard — donc les distributions ont probablement changé.

**Résultat sur notre dataset** : 1/40 features driftée (2.5% < seuil de 10%) → pas d'alerte. Le jeu de test ressemble suffisamment au jeu d'entraînement.

---

## 8. Le flux complet du projet

```
┌─────────────────────────────────────────────────────────────────┐
│                        DONNÉES BRUTES                           │
│         data/raw/train.csv  (137 restaurants + revenue)         │
│         data/raw/test.csv   (100 000 restaurants)               │
└──────────────────────────────┬──────────────────────────────────┘
                               │  dvc repro : étape load_data
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING                             │
│  • Calcul days_since_open                                        │
│  • Flags Px_active                                              │
│  • Suppression Id, Open Date                                    │
│  → data/processed/train_processed.csv                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │  dvc repro : étape preprocess
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  NOTEBOOK MODÉLISATION                          │
│  Nested CV (5 folds externes × 3 folds internes)               │
│  + Optuna TPE pour chaque fold                                  │
│  + 5 modèles testés (Ridge, Lasso, RF, XGBoost, LightGBM)       │
│  → 10 runs loggués dans MLflow                                  │
│  → RandomForest retenu (RMSE OOF = 2 508 158 TRY)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │  dvc repro : étape train
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│               ENTRAÎNEMENT FINAL (train.py)                     │
│  RandomForest entraîné sur les 137 restaurants complets         │
│  → models/best_model.pkl                                        │
│  → Run "Final_RandomForest" dans MLflow                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┼────────────────┐
               ▼               ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
    │  FastAPI     │  │  Dashboard   │  │  DVC evaluate    │
    │  /predict    │  │  Monitoring  │  │  metrics.json    │
    │  /health     │  │  (Streamlit) │  │  (RMSE, MAE, R²) │
    │  /docs       │  │              │  │                  │
    └──────────────┘  └──────────────┘  └──────────────────┘
           │
    ┌──────┴──────┐
    │   Docker    │
    │  (image     │
    │  portable)  │
    └─────────────┘
```

---

## 9. Questions jury probables et réponses

---

**"Pourquoi vous avez choisi la nested CV plutôt qu'une simple cross-validation ?"**

La cross-validation simple a un biais si on l'utilise pour optimiser ET évaluer en même temps. Optuna teste des dizaines de configurations d'hyperparamètres sur les mêmes folds — le modèle finit par être adapté à ces folds précis, et la métrique mesurée est trop optimiste. La nested CV isole complètement l'optimisation (boucle interne) de l'évaluation (boucle externe). Les 27 restaurants du fold externe n'ont jamais influencé le choix des hyperparamètres — leur RMSE est donc honnête.

---

**"R² = 0.045, c'est pas un peu faible ?"**

Non, c'est cohérent avec les données. La corrélation maximale entre une feature et le revenu est r = 0.192. Pour un modèle linéaire parfait, R² max = r² = 0.037. Ridge et Lasso, qui sont des modèles linéaires, ont effectivement des R² négatifs — ils font pire que prédire la moyenne. RandomForest à 0.045 dépasse ce plafond parce qu'il capte des relations non-linéaires. Mais avec des features aussi faiblement corrélées et seulement 137 observations, aucun modèle ne peut faire mieux. Ce résultat est cohérent avec les meilleures solutions de la compétition Kaggle originale.

---

**"Qu'est-ce qui se passe concrètement quand quelqu'un appelle l'API ?"**

1. La requête JSON arrive sur `POST /predict`
2. Pydantic vérifie que tous les champs sont présents et ont le bon type
3. Le `RestaurantInput` est converti en DataFrame pandas
4. Le pipeline sklearn applique les transformations (calcul `days_since_open`, StandardScaler, OneHotEncoder) avec les paramètres appris à l'entraînement
5. RandomForest prédit `log(revenue)`
6. `exp(prédiction) - 1` est appliqué pour revenir en TRY
7. La réponse JSON `{"prediction": 4690480.8}` est retournée

---

**"Pourquoi DVC et pas juste un Makefile ?"**

Un Makefile peut aussi orchestrer des étapes. La différence : DVC calcule des hash cryptographiques de chaque fichier d'entrée et de sortie. Si `train_processed.csv` n'a pas changé depuis la dernière exécution, `dvc repro` ne relance pas `train.py`. Un Makefile vérifie les dates de modification des fichiers — moins fiable. De plus, DVC gère le versioning des données volumineuses (non stockées dans git) et peut les synchroniser avec un stockage distant (S3, GCS).

---

**"Comment vous savez que votre API tient la charge ?"**

Le projet actuel ne fait pas de tests de charge. Pour aller plus loin en production, on utiliserait des outils comme Locust ou k6. Ce qui est en place : le modèle est chargé une seule fois en mémoire au démarrage (pas à chaque requête), et le Docker conteneurise l'application pour pouvoir lancer plusieurs instances en parallèle.

---

**"Pourquoi RandomForest plutôt que XGBoost, la différence est très faible ?"**

L'écart est de 7 600 TRY de RMSE — c'est 0.3% de différence. Les deux modèles sont statistiquement équivalents sur ce dataset. RandomForest a été retenu car Optuna a convergé vers de meilleurs hyperparamètres en nested CV. Sur d'autres splits ou d'autres seeds, XGBoost pourrait gagner. En production, on choisirait probablement les deux en ensemble pour moyenner leurs prédictions.

---

**"C'est quoi la différence entre le run 'NestedCV_Optuna_RandomForest' et 'Final_RandomForest' dans MLflow ?"**

`NestedCV_Optuna_RandomForest` est le run d'évaluation : il mesure honnêtement la performance via nested CV (RMSE OOF = 2 508 158). Le modèle final qui y est archivé est entraîné sur **tout le dataset** avec les meilleurs hyperparamètres trouvés — il n'a pas été évalué, juste entraîné.

`Final_RandomForest` est le run de production : le même modèle final, loggué via `train.py` dans le pipeline DVC, sauvegardé dans `models/best_model.pkl` et servi par l'API.

---

**"Qu'est-ce qui déclencherait un réentraînement du modèle ?"**

Deux signaux principaux :
1. Le dashboard de monitoring détecte que plus de 10% des features dérivent significativement — le modèle reçoit des données qui ne ressemblent plus à ses données d'entraînement.
2. La performance mesurée sur des données labellisées récentes (si TFI communique les vrais revenus de nouveaux restaurants) se dégrade significativement.

Dans les deux cas, on relancerait `dvc repro` avec les nouvelles données intégrées dans le dataset d'entraînement.
