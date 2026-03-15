"""
train.py
---------
Trains XGBoost, LightGBM, CatBoost with Optuna tuning.
Logs everything to MLflow. Saves best model.

Usage:
    python src/train.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report, average_precision_score
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# ── KS Statistic ──────────────────────────────────────────────────────────────

def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov stat — standard credit risk metric."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
    df["cum_pos"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_neg"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return (df["cum_pos"] - df["cum_neg"]).abs().max()


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data/processed"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Default rate — Train: {y_train.mean():.2%}  Test: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ── SMOTE balancing ───────────────────────────────────────────────────────────

def apply_smote(X, y, random_state=42):
    print("Applying SMOTE ...")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  After SMOTE: {X_res.shape}, default rate: {y_res.mean():.2%}")
    return X_res, y_res


# ── Optuna objectives ─────────────────────────────────────────────────────────

def xgb_objective(trial, X, y, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1000),
        "max_depth":         trial.suggest_int("max_depth", 4, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "use_label_encoder": False,
        "eval_metric":       "auc",
        "random_state":      42,
        "n_jobs":            1,
        "tree_method":       "hist",
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()


def lgb_objective(trial, X, y, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 300, 1000),
        "max_depth":         trial.suggest_int("max_depth", 4, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 31, 200),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "is_unbalance":      True,
        "random_state":      42,
        "n_jobs":            1,
        "verbosity":         -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()


def catboost_objective(trial, X, y, cv):
    params = {
        "iterations":        trial.suggest_int("iterations", 300, 800),
        "depth":             trial.suggest_int("depth", 4, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "auto_class_weights": "Balanced",
        "random_seed":       42,
        "verbose":           0,
    }
    model = CatBoostClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return scores.mean()


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_test, y_prob), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "ks_stat":   round(ks_statistic(y_test.values, y_prob), 4),
    }

    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def run(data_dir: str = "data/processed", model_dir: str = "models", n_trials: int = 30):
    os.makedirs(model_dir, exist_ok=True)
    mlflow.set_experiment("credit_risk_causal_ml")

    X_train, X_test, y_train, y_test = load_data(data_dir)
    if len(X_train) > 100_000:
        print(f"Sampling down from {len(X_train)} to 100,000 rows for speed ...")
        sample_idx = X_train.sample(100_000, random_state=42).index
        X_train = X_train.loc[sample_idx]
        y_train = y_train.loc[sample_idx]
        print(f"Sampled train size: {X_train.shape}")
    X_res, y_res = apply_smote(X_train, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall = {"auc_roc": 0}

    model_configs = [
        ("XGBoost",   xgb_objective,      xgb.XGBClassifier),
        ("LightGBM",  lgb_objective,      lgb.LGBMClassifier),
        ("CatBoost",  catboost_objective,  CatBoostClassifier),
    ]

    for model_name, objective, ModelClass in model_configs:
        print(f"\n{'='*55}")
        print(f"  Tuning {model_name} ({n_trials} Optuna trials) ...")
        print(f"{'='*55}")

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda t: objective(t, X_res, y_res, cv),
                       n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        print(f"  Best CV AUC-ROC: {study.best_value:.4f}")
        print(f"  Best params: {best_params}")

        # Refit on full SMOTE data with best params
        if model_name == "CatBoost":
            best_params.update({"auto_class_weights": "Balanced",
                                 "random_seed": 42, "verbose": 0})
            model = CatBoostClassifier(**best_params)
        elif model_name == "LightGBM":
            best_params.update({"random_state": 42, "n_jobs": -1,
                                 "verbosity": -1, "is_unbalance": True})
            model = lgb.LGBMClassifier(**best_params)
        else:
            best_params.update({"use_label_encoder": False, "eval_metric": "auc",
                                 "random_state": 42, "n_jobs": -1, "tree_method": "hist"})
            model = xgb.XGBClassifier(**best_params)

        model.fit(X_res, y_res)
        metrics = evaluate(model, X_test, y_test, model_name)

        # Log to MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=model_name)

        # Save model artifact
        joblib.dump(model, f"{model_dir}/{model_name.lower()}_model.pkl")
        joblib.dump(list(X_train.columns), f"{model_dir}/feature_names.pkl")

        if metrics["auc_roc"] > best_overall["auc_roc"]:
            best_overall = {**metrics, "model_name": model_name, "model": model}

    # Save best model as production model
    best_model = best_overall.pop("model")
    print(f"\n🏆 Best model: {best_overall['model_name']}  "
          f"AUC-ROC={best_overall['auc_roc']}  KS={best_overall['ks_stat']}")
    joblib.dump(best_model, f"{model_dir}/best_model.pkl")
    pd.DataFrame([best_overall]).to_csv(f"{model_dir}/best_metrics.csv", index=False)
    print(f"\n✅ Training complete. Best model saved → {model_dir}/best_model.pkl")


if __name__ == "__main__":
    run()
