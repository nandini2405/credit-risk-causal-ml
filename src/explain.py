"""
explain.py
-----------
Generates SHAP global + local explanations.
Saves plots to models/plots/.

Usage:
    python src/explain.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_artifacts(model_dir: str = "models", data_dir: str = "data/processed"):
    model = joblib.load(f"{model_dir}/best_model.pkl")
    feature_names = joblib.load(f"{model_dir}/feature_names.pkl")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    return model, feature_names, X_test, y_test


def get_explainer(model, X_sample):
    """Auto-select best SHAP explainer for the model type."""
    model_type = type(model).__name__
    if "XGB" in model_type or "LGBM" in model_type or "CatBoost" in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_sample)
    return explainer


def plot_summary(shap_values, X_sample, out_dir: str):
    """Beeswarm summary plot — global feature importance."""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=(12, 8))
    plt.title("SHAP Summary — Global Feature Impact on Default Risk", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/shap_summary.png")


def plot_bar(shap_values, X_sample, out_dir: str):
    """Bar plot — mean absolute SHAP values."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("Mean |SHAP| — Feature Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/shap_bar.png")


def plot_waterfall_local(explainer, shap_values, X_sample, idx: int, out_dir: str):
    """Waterfall plot — single applicant explanation."""
    plt.figure(figsize=(12, 6))
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1])
    else:
        base_val = float(base_val)

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[idx],
            base_values=base_val,
            data=X_sample.iloc[idx],
            feature_names=list(X_sample.columns)
        ),
        show=False
    )
    plt.title(f"SHAP Waterfall — Applicant #{idx} (Local Explanation)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/shap_waterfall_idx{idx}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_dir}/shap_waterfall_idx{idx}.png")


def get_top_shap_factors(shap_values, feature_names, idx: int, n: int = 3) -> list[dict]:
    """Return top-n causal SHAP factors for a single prediction (used in API)."""
    sv = shap_values[idx]
    sorted_idx = np.argsort(np.abs(sv))[::-1][:n]
    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(sv[i]), 4),
            "direction": "increases risk" if sv[i] > 0 else "decreases risk"
        }
        for i in sorted_idx
    ]


def run(model_dir: str = "models", data_dir: str = "data/processed",
        sample_size: int = 5000):
    out_dir = f"{model_dir}/plots"
    os.makedirs(out_dir, exist_ok=True)

    print("=== SHAP Explainability ===\n")

    print("[1/4] Loading artifacts ...")
    model, feature_names, X_test, y_test = load_artifacts(model_dir, data_dir)

    # Sample for speed
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    print(f"  Using {len(X_sample)} test samples for SHAP computation.")

    print("[2/4] Computing SHAP values ...")
    explainer = get_explainer(model, X_sample)
    shap_values_raw = explainer.shap_values(X_sample)

    # Handle binary classifiers that return 2D list
    if isinstance(shap_values_raw, list):
        shap_values = shap_values_raw[1]   # class=1 (default)
    else:
        shap_values = shap_values_raw

    joblib.dump({"explainer": explainer,
                 "shap_values": shap_values,
                 "X_sample": X_sample},
                f"{model_dir}/shap_artifacts.pkl")
    print(f"  SHAP artifacts saved → {model_dir}/shap_artifacts.pkl")

    print("[3/4] Generating global plots ...")
    plot_summary(shap_values, X_sample, out_dir)
    plot_bar(shap_values, X_sample, out_dir)

    print("[4/4] Generating local waterfall plots (high risk + low risk) ...")
    y_prob = model.predict_proba(X_sample)[:, 1]

    # Pick one high-risk and one low-risk applicant
    high_risk_idx = int(np.argmax(y_prob))
    low_risk_idx  = int(np.argmin(y_prob))
    plot_waterfall_local(explainer, shap_values, X_sample, high_risk_idx, out_dir)
    plot_waterfall_local(explainer, shap_values, X_sample, low_risk_idx, out_dir)

    # Print top factors for a sample applicant
    print("\n  Sample — Top 3 SHAP factors for highest-risk applicant:")
    for f in get_top_shap_factors(shap_values, list(X_sample.columns), high_risk_idx):
        print(f"    {f['feature']:25s} → {f['shap_value']:+.4f}  ({f['direction']})")

    print("\n✅ Explainability complete. Plots saved to", out_dir)


if __name__ == "__main__":
    run()
