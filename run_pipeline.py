"""
run_pipeline.py
----------------
Master script — runs the full pipeline end-to-end.

Usage:
    python run_pipeline.py --input data/lending_club_raw.csv
    python run_pipeline.py --input data/lending_club_raw.csv --trials 50
"""

import argparse
import subprocess
import sys
import os


def run_step(description: str, command: list[str]):
    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print(f"{'='*60}")
    result = subprocess.run(command, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Step failed: {description}")
        sys.exit(1)
    print(f"\n✅ Done: {description}")


def main():
    parser = argparse.ArgumentParser(description="Credit Risk Causal ML — Full Pipeline")
    parser.add_argument("--input",    default="data/lending_club_raw.csv",
                        help="Path to raw LendingClub CSV")
    parser.add_argument("--out_dir",  default="data/processed")
    parser.add_argument("--model_dir",default="models")
    parser.add_argument("--trials",   type=int, default=30,
                        help="Optuna trials per model (higher = better, slower)")
    parser.add_argument("--skip_causal", action="store_true",
                        help="Skip causal analysis (faster for testing)")
    args = parser.parse_args()

    print("\n🏦  Credit Risk Causal ML — Full Pipeline")
    print(f"    Input: {args.input}")
    print(f"    Output: {args.out_dir}  |  Models: {args.model_dir}")
    print(f"    Optuna trials: {args.trials}")

    # 1. Preprocess
    run_step("Data Preprocessing", [
        sys.executable, "src/preprocess.py",
        "--input", args.input,
        "--out_dir", args.out_dir,
    ])

    # 2. Causal analysis (optional)
    if not args.skip_causal:
        run_step("Causal Graph + ATE Estimation", [
            sys.executable, "src/causal_model.py",
        ])
    else:
        print("\n⏭️  Skipping causal analysis (--skip_causal set)")

    # 3. Train models
    run_step("Model Training (XGBoost + LightGBM + CatBoost + Optuna)", [
        sys.executable, "src/train.py",
    ])

    # 4. Explainability
    run_step("SHAP Explainability", [
        sys.executable, "src/explain.py",
    ])

    print("\n" + "="*60)
    print("  🎉  PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Start API:       uvicorn api.main:app --reload --port 8000")
    print("  2. Start dashboard: streamlit run app.py")
    print("  3. View MLflow UI:  mlflow ui  (open http://localhost:5000)")
    print("  4. Docker deploy:   docker-compose up --build")
    print("="*60)


if __name__ == "__main__":
    main()
