"""
causal_model.py
----------------
Builds a causal DAG for credit default using DoWhy.
Estimates Average Treatment Effects (ATE) for key financial variables.

Usage:
    python src/causal_model.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print("⚠️  DoWhy not installed. Run: pip install dowhy")


# ── Causal Graph Definition ──────────────────────────────────────────────────

# Domain-knowledge DAG:
#  income → dti → default
#  fico_score → int_rate → default
#  emp_length → loan_amnt → default
#  dti → default
#  int_rate → default
#  loan_to_income → default

CAUSAL_GRAPH = """
digraph {
    annual_inc -> dti;
    annual_inc -> loan_to_income;
    annual_inc -> default;
    dti -> default;
    fico_score -> int_rate;
    fico_score -> default;
    int_rate -> default;
    emp_length -> loan_amnt;
    loan_amnt -> loan_to_income;
    loan_to_income -> default;
    payment_to_income -> default;
    credit_utilization -> default;
    delinq_2yrs -> default;
    pub_rec -> default;
}
"""


# ── Visualise DAG ─────────────────────────────────────────────────────────────

def plot_causal_dag(save_path: str = "models/causal_dag.png"):
    """Draw and save the causal DAG."""
    G = nx.DiGraph()
    edges = [
        ("annual_inc", "dti"), ("annual_inc", "loan_to_income"), ("annual_inc", "default"),
        ("dti", "default"), ("fico_score", "int_rate"), ("fico_score", "default"),
        ("int_rate", "default"), ("emp_length", "loan_amnt"),
        ("loan_amnt", "loan_to_income"), ("loan_to_income", "default"),
        ("payment_to_income", "default"), ("credit_utilization", "default"),
        ("delinq_2yrs", "default"), ("pub_rec", "default"),
    ]
    G.add_edges_from(edges)

    color_map = []
    for node in G.nodes():
        if node == "default":
            color_map.append("#e74c3c")  # red = outcome
        elif node in ("dti", "int_rate", "loan_to_income", "payment_to_income", "credit_utilization"):
            color_map.append("#e67e22")  # orange = mediators
        else:
            color_map.append("#2980b9")  # blue = causes

    plt.figure(figsize=(14, 8))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx(
        G, pos,
        node_color=color_map,
        node_size=2200,
        font_size=9,
        font_color="white",
        font_weight="bold",
        edge_color="#555",
        arrows=True,
        arrowsize=20,
        width=1.8,
    )
    plt.title("Causal DAG — Credit Default Risk", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  DAG saved → {save_path}")


# ── ATE Estimation ────────────────────────────────────────────────────────────

def estimate_ate(df: pd.DataFrame, treatment: str, outcome: str = "default") -> dict:
    """
    Estimate Average Treatment Effect of `treatment` on `outcome`
    using DoWhy backdoor adjustment.
    Returns dict with ate, p_value, interpretation.
    """
    if not DOWHY_AVAILABLE:
        return {"error": "DoWhy not available"}

    # Binarize treatment at median for ATE estimation
    median_val = df[treatment].median()
    df = df.copy()
    df[f"{treatment}_binary"] = (df[treatment] > median_val).astype(int)
    treatment_col = f"{treatment}_binary"

    # Build minimal graph for this treatment
    common_causes = [c for c in ["annual_inc", "fico_score", "emp_length"]
                     if c in df.columns and c != treatment]

    try:
        model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome,
            common_causes=common_causes,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
            target_units="ate",
        )
        ate_val = float(estimate.value)
        return {
            "treatment": treatment,
            "threshold": f"> median ({median_val:.2f})",
            "ATE": round(ate_val, 4),
            "interpretation": (
                f"Having {treatment} above its median {'increases' if ate_val > 0 else 'decreases'} "
                f"default probability by {abs(ate_val)*100:.2f} percentage points (causal estimate)."
            )
        }
    except Exception as e:
        return {"treatment": treatment, "error": str(e)}


def run_causal_analysis(data_path: str = "data/processed/X_train.csv",
                        target_path: str = "data/processed/y_train.csv",
                        out_dir: str = "models"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("=== Causal Analysis ===\n")

    # 1. Plot DAG
    print("[1/3] Plotting causal DAG ...")
    plot_causal_dag(f"{out_dir}/causal_dag.png")

    # 2. Load data
    print("[2/3] Loading processed data ...")
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path).squeeze()
    df = X.copy()
    df["default"] = y.values

    # Use a sample for speed
    if len(df) > 50_000:
        df = df.sample(50_000, random_state=42)
        print(f"  Using sample of 50,000 rows for causal estimation.")

    # 3. Estimate ATE for key treatments
    print("[3/3] Estimating Average Treatment Effects ...")
    treatments = ["dti", "int_rate", "fico_score", "loan_to_income",
                  "credit_utilization", "delinq_2yrs"]
    treatments = [t for t in treatments if t in df.columns]

    results = []
    for t in treatments:
        res = estimate_ate(df, treatment=t)
        results.append(res)
        if "ATE" in res:
            print(f"  {t:25s} → ATE = {res['ATE']:+.4f}")
            print(f"    {res['interpretation']}")
        else:
            print(f"  {t:25s} → {res.get('error', 'unknown error')}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{out_dir}/causal_ate_results.csv", index=False)
    print(f"\n  ATE results saved → {out_dir}/causal_ate_results.csv")
    print("\n✅ Causal analysis complete.")
    return results_df


if __name__ == "__main__":
    run_causal_analysis()
