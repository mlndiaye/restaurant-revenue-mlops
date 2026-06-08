"""
Experiment utilities for model selection notebooks.

The notebook should stay focused on the modelling story: load data, choose a
protocol, run experiments, inspect results. This module contains the reusable
Optuna and nested cross-validation mechanics.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import warnings

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from restaurant_revenue.features.preprocessing import get_preprocessor_pipeline


RANDOM_STATE = 42
DEFAULT_N_TRIALS_BY_MODEL = {
    "Ridge": 25,
    "Lasso": 25,
    "RandomForest": 30,
    "XGBoost": 30,
    "LightGBM": 30,
}


def _suppress_expected_model_warnings() -> None:
    """Hide noisy numerical warnings from unstable trial candidates."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def make_cv(n_splits: int, random_state: int = RANDOM_STATE) -> KFold:
    """Create a reproducible K-Fold splitter."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def rmse_score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE on the original revenue scale."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_base_model(model_name: str, params: dict[str, Any], random_state: int = RANDOM_STATE):
    """Instantiate a model from an Optuna parameter dictionary."""
    if model_name == "Ridge":
        return Ridge(alpha=params["alpha"])

    if model_name == "Lasso":
        return Lasso(alpha=params["alpha"], max_iter=30000, random_state=random_state)

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=random_state,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            min_child_weight=params["min_child_weight"],
            random_state=random_state,
            n_jobs=1,
        )

    if model_name == "LightGBM":
        return LGBMRegressor(
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            min_child_samples=params["min_child_samples"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def preprocessing_from_params(model_name: str, params: dict[str, Any]) -> dict[str, bool]:
    """Choose preprocessing switches from model family and Optuna params."""
    return {
        "scale_numeric": model_name in {"Ridge", "Lasso"},
        "add_sparse_flags": params["add_sparse_flags"],
    }


def make_estimator(
    model_name: str,
    params: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> TransformedTargetRegressor:
    """Build the full anti-leakage estimator.

    The target transformation lives inside TransformedTargetRegressor so CV
    scores predictions on the original revenue scale while fitting on log1p(y).
    """
    preprocessor = get_preprocessor_pipeline(**preprocessing_from_params(model_name, params))
    feature_model_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", clone(build_base_model(model_name, params, random_state=random_state))),
        ]
    )
    return TransformedTargetRegressor(
        regressor=feature_model_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def suggest_params(model_name: str, trial: optuna.Trial) -> dict[str, Any]:
    """Define Optuna search spaces by model family."""
    params: dict[str, Any] = {
        "add_sparse_flags": trial.suggest_categorical("add_sparse_flags", [False, True]),
    }

    if model_name == "Ridge":
        params["alpha"] = trial.suggest_float("alpha", 1e-2, 1e3, log=True)

    elif model_name == "Lasso":
        params["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)

    elif model_name == "RandomForest":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, None]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", [1.0, "sqrt", "log2"]),
            }
        )

    elif model_name == "XGBoost":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
        )

    elif model_name == "LightGBM":
        params.update(
            {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return params


def objective(
    trial: optuna.Trial,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_cv: KFold,
    random_state: int = RANDOM_STATE,
) -> float:
    """Optuna objective evaluated by internal cross-validation."""
    params = suggest_params(model_name, trial)
    fold_scores = []

    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train, y_train), start=1):
        X_inner_train = X_train.iloc[inner_train_idx]
        X_inner_val = X_train.iloc[inner_val_idx]
        y_inner_train = y_train.iloc[inner_train_idx]
        y_inner_val = y_train.iloc[inner_val_idx]

        estimator = make_estimator(model_name, params, random_state=random_state)
        with warnings.catch_warnings():
            _suppress_expected_model_warnings()
            estimator.fit(X_inner_train, y_inner_train)
            preds = estimator.predict(X_inner_val)

        fold_scores.append(rmse_score(y_inner_val, preds))
        trial.report(float(np.mean(fold_scores)), step=inner_fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


def run_optuna_study(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    inner_cv: KFold,
    n_trials: int,
    random_state: int = RANDOM_STATE,
    seed_offset: int = 0,
) -> optuna.Study:
    """Run one Optuna study for a model on the current training fold."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=random_state + seed_offset)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train, inner_cv, random_state=random_state),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study


def fit_best_estimator(
    model_name: str,
    best_params: dict[str, Any],
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    random_state: int = RANDOM_STATE,
) -> TransformedTargetRegressor:
    """Fit the selected estimator on a complete train split."""
    estimator = make_estimator(model_name, best_params, random_state=random_state)
    with warnings.catch_warnings():
        _suppress_expected_model_warnings()
        estimator.fit(X_fit, y_fit)
    return estimator


def trial_history(study: optuna.Study) -> list[dict[str, Any]]:
    """Serialize an Optuna study history for MLflow logging."""
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "params": trial.params,
            }
        )
    return rows


def _delete_runs_by_name(run_name: str) -> None:
    """Delete all MLflow runs with run_name across all active experiments."""
    client = mlflow.tracking.MlflowClient()
    for exp in client.search_experiments():
        for run in client.search_runs(experiment_ids=[exp.experiment_id]):
            if run.info.run_name == run_name:
                client.delete_run(run.info.run_id)
                print(f"  [idempotent] Deleted existing run '{run_name}' ({run.info.run_id[:8]}...)")


def run_nested_optuna_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str] | None = None,
    n_trials_by_model: dict[str, int] | None = None,
    outer_cv: KFold | None = None,
    inner_cv: KFold | None = None,
    random_state: int = RANDOM_STATE,
    log_to_mlflow: bool = True,
    overwrite: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run nested CV with Optuna for all selected models."""
    n_trials_by_model = n_trials_by_model or DEFAULT_N_TRIALS_BY_MODEL
    model_names = model_names or list(n_trials_by_model)
    outer_cv = outer_cv or make_cv(n_splits=5, random_state=random_state)
    inner_cv = inner_cv or make_cv(n_splits=3, random_state=random_state)

    results: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        oof_preds = np.zeros(len(X))
        fold_reports = []

        if log_to_mlflow and overwrite:
            _delete_runs_by_name(f"NestedCV_Optuna_{model_name}")

        run_context = mlflow.start_run(run_name=f"NestedCV_Optuna_{model_name}") if log_to_mlflow else nullcontext()
        with run_context:
            if log_to_mlflow:
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("search_strategy", "optuna_tpe_nested_cv")
                mlflow.log_param("outer_cv", outer_cv.get_n_splits())
                mlflow.log_param("inner_cv", inner_cv.get_n_splits())
                mlflow.log_param("n_trials", n_trials_by_model[model_name])
                mlflow.log_param("sampler", "TPESampler")
                mlflow.log_param("pruner", "MedianPruner")

            for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                study = run_optuna_study(
                    model_name,
                    X_train,
                    y_train,
                    inner_cv,
                    n_trials=n_trials_by_model[model_name],
                    random_state=random_state,
                    seed_offset=fold,
                )
                best_params = study.best_trial.params
                best_estimator = fit_best_estimator(model_name, best_params, X_train, y_train, random_state=random_state)

                with warnings.catch_warnings():
                    _suppress_expected_model_warnings()
                    preds = best_estimator.predict(X_val)
                oof_preds[val_idx] = preds

                fold_rmse = rmse_score(y_val, preds)
                fold_mae = mean_absolute_error(y_val, preds)
                fold_r2 = r2_score(y_val, preds)

                report = {
                    "fold": fold,
                    "inner_rmse": study.best_value,
                    "outer_rmse": fold_rmse,
                    "outer_mae": fold_mae,
                    "outer_r2": fold_r2,
                    "best_params": best_params,
                    "n_complete_trials": len(
                        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                    ),
                }
                fold_reports.append(report)

                print(
                    f"Fold {fold} | inner RMSE={study.best_value:,.0f} "
                    f"| outer RMSE={fold_rmse:,.0f} | params={best_params}"
                )

            rmse = rmse_score(y, oof_preds)
            mae = mean_absolute_error(y, oof_preds)
            r2 = r2_score(y, oof_preds)

            final_study = run_optuna_study(
                model_name,
                X,
                y,
                inner_cv,
                n_trials=n_trials_by_model[model_name],
                random_state=random_state,
                seed_offset=999,
            )
            final_params = final_study.best_trial.params
            final_estimator = fit_best_estimator(model_name, final_params, X, y, random_state=random_state)

            results[model_name] = {
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "predictions": oof_preds,
                "fold_reports": fold_reports,
                "final_inner_rmse": final_study.best_value,
                "final_best_params": final_params,
                "final_estimator": final_estimator,
            }

            if log_to_mlflow:
                mlflow.log_metric("oof_rmse", rmse)
                mlflow.log_metric("oof_mae", mae)
                mlflow.log_metric("oof_r2", r2)
                mlflow.log_metric("final_inner_cv_rmse", final_study.best_value)
                mlflow.log_params({f"final_{k}": v for k, v in final_params.items()})
                mlflow.log_dict(fold_reports, "fold_optuna_results.json")
                mlflow.log_dict(trial_history(final_study), "final_study_trials.json")
                mlflow.sklearn.log_model(final_estimator, name="model")

            print(f"{model_name} OOF -> RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | R2: {r2:.4f}")
            print(f"Final full-data Optuna -> inner RMSE: {final_study.best_value:,.2f} | params: {final_params}")

    return results


def results_to_dataframe(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Convert experiment results to a notebook-friendly summary table."""
    return pd.DataFrame(
        {
            name: {
                "RMSE": res["RMSE"],
                "MAE": res["MAE"],
                "R2": res["R2"],
                "Final inner CV RMSE": res["final_inner_rmse"],
                "Final params": res["final_best_params"],
            }
            for name, res in results.items()
        }
    ).T.sort_values("RMSE")
