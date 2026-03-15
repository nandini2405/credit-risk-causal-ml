"""
preprocess.py
--------------
Loads raw LendingClub data, cleans it, engineers features,
and saves train/test splits ready for modeling.

Usage:
    python src/preprocess.py --input data/lending_club_raw.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")


# ── Column subsets ──────────────────────────────────────────────────────────

KEEP_COLS = [
    "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state",
    "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "initial_list_status",
    "application_type", "loan_status"
]

TARGET_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
    "In Grace Period": 1,
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def clean_term(val):
    """' 36 months' → 36"""
    if pd.isnull(val):
        return np.nan
    return int(str(val).strip().split()[0])


def clean_int_rate(val):
    """'13.56%' → 13.56"""
    if pd.isnull(val):
        return np.nan
    return float(str(val).replace("%", "").strip())


def clean_revol_util(val):
    """'62.5%' → 62.5"""
    if pd.isnull(val):
        return np.nan
    return float(str(val).replace("%", "").strip())


def clean_emp_length(val):
    """'10+ years' → 10, '< 1 year' → 0, 'n/a' → NaN"""
    if pd.isnull(val):
        return np.nan
    val = str(val).strip().lower()
    if val in ("n/a", "nan", ""):
        return np.nan
    if "< 1" in val:
        return 0
    if "10+" in val:
        return 10
    return int("".join(filter(str.isdigit, val)))


# ── Main pipeline ────────────────────────────────────────────────────────────

def load_and_filter(path: str) -> pd.DataFrame:
    print(f"[1/6] Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"      Raw shape: {df.shape}")

    # Keep only relevant columns that exist
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # Keep only rows with a mappable loan_status
    df = df[df["loan_status"].isin(TARGET_MAP.keys())].copy()
    df["default"] = df["loan_status"].map(TARGET_MAP)
    df.drop(columns=["loan_status"], inplace=True)

    print(f"      After filter shape: {df.shape}")
    print(f"      Default rate: {df['default'].mean():.2%}")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Cleaning columns ...")
    df["term"] = df["term"].apply(clean_term)
    df["int_rate"] = df["int_rate"].apply(clean_int_rate)
    df["revol_util"] = df["revol_util"].apply(clean_revol_util)
    df["emp_length"] = df["emp_length"].apply(clean_emp_length)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Engineering features ...")

    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)

    df["payment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1e-6)
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1e-6)
    df["credit_utilization"] = df["revol_bal"] / (df["annual_inc"] + 1e-6)

    # Grade → ordinal
    grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_num"] = df["grade"].map(grade_order)

    df.drop(columns=["sub_grade", "addr_state", "grade"], inplace=True, errors="ignore")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("[4/6] Encoding categoricals ...")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/6] Handling missing values ...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def split_and_save(df: pd.DataFrame, encoders: dict, out_dir: str):
    print("[6/6] Splitting and saving ...")
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)
    joblib.dump(encoders, f"{out_dir}/encoders.pkl")

    print(f"      Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"      Saved to {out_dir}/")
    return X_train, X_test, y_train, y_test


def run(input_path: str, out_dir: str = "data/processed"):
    df = load_and_filter(input_path)
    df = clean_columns(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)
    df = handle_missing(df)
    split_and_save(df, encoders, out_dir)
    print("\n✅ Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/lending_club_raw.csv")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()
    run(args.input, args.out_dir)
