"""
Hyperparameter Tuning for HistGradientBoosting Win Probability Model
====================================================================

Purpose:
- Tune HistGradientBoostingClassifier after model comparison selected it as the main model.
- Use stratified 5-fold cross-validation.
- Stratification tries to preserve product_model + success/fail distribution.
- Tune with RandomizedSearchCV using multiple metrics.

Recommended run from project root:
    python src/surya_hgb_hyperparameter_tuning.py --data dataset/df_preprocessed.csv

If this file is in the current folder:
    python surya_hgb_hyperparameter_tuning.py --data dataset/df_preprocessed.csv

Main outputs in output/:
- hgb_tuning_cv_results.csv
- hgb_best_params.csv
- hgb_best_params.txt
- hgb_tuned_5fold_results.csv
- hgb_tuned_5fold_summary.csv
- final_tuned_hgb_predictions.csv
- final_tuned_hgb_model.joblib
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Default configuration
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
MIN_PRODUCT_COUNT = 15
OUTPUT_DIR = "output"

FEATURES = [
    # basic quote information
    "product_model",
    "kw",
    "unit_price",
    "qty",
    "subtotal_price",
    "gross_margin_rate",
    "energy_grant_amount",

    # competitor raw prices
    "competitor_a",
    "competitor_b",
    "competitor_c",

    # competitor availability
    "is_compe_a",
    "is_compe_b",
    "is_compe_c",
    "competitor_count_available",

    # competitor summary
    "avg_competitor_price",
    "min_competitor_price",
    "max_competitor_price",

    # competitor positioning
    "price_order",
    "price_gap_avg_competitor",
    "price_gap_avg_competitor_pct",
    "price_gap_min_competitor",
    "price_gap_min_competitor_pct",
    "higher_than_avg_competitor",
    "is_lower_than_competitor",

    # grant and profit
    "effective_price_after_grant",
    "grant_ratio_to_subtotal",
    "estimated_cost",
    "estimated_gross_profit",
]

TARGET = "success"
CATEGORICAL_FEATURES = ["product_model"]

SCORING_MAP = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "neg_log_loss": "neg_log_loss",
    "neg_brier_score": "neg_brier_score",
}


@dataclass
class CVOutput:
    fold_results: pd.DataFrame
    summary: pd.DataFrame


# ============================================================
# Data loading and preparation
# ============================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load preprocessed quotation data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please check the path. Example from project root: "
            "--data dataset/df_preprocessed.csv"
        )

    df = pd.read_csv(data_path)

    # Handle older naming if the target column is still called Success.
    if "convert_to_order" not in df.columns and "Success" in df.columns:
        df = df.rename(columns={"Success": "convert_to_order"})

    if "convert_to_order" not in df.columns:
        raise ValueError(
            "Column 'convert_to_order' is required. "
            "Expected coding: 0 = Success, 1 = Fail."
        )

    return df


def add_missing_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add required helper features if they are not already available.
    This keeps the script compatible with both raw-ish and preprocessed files.
    """
    df = df.copy()

    competitor_cols = [
        col for col in ["competitor_a", "competitor_b", "competitor_c"]
        if col in df.columns
    ]

    # Competitor availability flags
    for source_col, flag_col in [
        ("competitor_a", "is_compe_a"),
        ("competitor_b", "is_compe_b"),
        ("competitor_c", "is_compe_c"),
    ]:
        if flag_col not in df.columns:
            if source_col in df.columns:
                df[flag_col] = df[source_col].notna().astype(int)
            else:
                df[flag_col] = 0

    if "competitor_count_available" not in df.columns:
        if competitor_cols:
            df["competitor_count_available"] = df[competitor_cols].notna().sum(axis=1)
        else:
            df["competitor_count_available"] = 0

    # Competitor summary prices
    if "avg_competitor_price" not in df.columns:
        df["avg_competitor_price"] = df[competitor_cols].mean(axis=1) if competitor_cols else np.nan

    if "min_competitor_price" not in df.columns:
        df["min_competitor_price"] = df[competitor_cols].min(axis=1) if competitor_cols else np.nan

    if "max_competitor_price" not in df.columns:
        df["max_competitor_price"] = df[competitor_cols].max(axis=1) if competitor_cols else np.nan

    # Price order within the same quote_id + product.
    # 1 means highest unit price within that duplicate group.
    if "price_order" not in df.columns:
        if {"quote_id", "product", "unit_price"}.issubset(df.columns):
            df["price_order"] = (
                df.groupby(["quote_id", "product"])["unit_price"]
                .rank(method="dense", ascending=False)
                .astype(int)
            )
        else:
            df["price_order"] = np.nan

    if "is_highest_price" not in df.columns:
        df["is_highest_price"] = (df["price_order"] == 1).astype(int)

    # Price gap vs competitor
    if "price_gap_avg_competitor" not in df.columns:
        df["price_gap_avg_competitor"] = df["unit_price"] - df["avg_competitor_price"]

    if "price_gap_avg_competitor_pct" not in df.columns:
        df["price_gap_avg_competitor_pct"] = np.where(
            df["avg_competitor_price"].notna() & (df["avg_competitor_price"] != 0),
            (df["price_gap_avg_competitor"] / df["avg_competitor_price"]) * 100,
            np.nan,
        )

    if "price_gap_min_competitor" not in df.columns:
        df["price_gap_min_competitor"] = df["unit_price"] - df["min_competitor_price"]

    if "price_gap_min_competitor_pct" not in df.columns:
        df["price_gap_min_competitor_pct"] = np.where(
            df["min_competitor_price"].notna() & (df["min_competitor_price"] != 0),
            (df["price_gap_min_competitor"] / df["min_competitor_price"]) * 100,
            np.nan,
        )

    if "higher_than_avg_competitor" not in df.columns:
        df["higher_than_avg_competitor"] = (
            df["price_gap_avg_competitor"] > 0
        ).astype(int)

    if "is_lower_than_competitor" not in df.columns:
        df["is_lower_than_competitor"] = (
            df["price_gap_min_competitor"] < 0
        ).astype(int)

    # Grant and profit features
    if "energy_grant_amount" in df.columns:
        energy_grant = df["energy_grant_amount"].fillna(0)
    else:
        df["energy_grant_amount"] = 0
        energy_grant = df["energy_grant_amount"]

    if "effective_price_after_grant" not in df.columns:
        df["effective_price_after_grant"] = df["subtotal_price"] - energy_grant

    if "grant_ratio_to_subtotal" not in df.columns:
        df["grant_ratio_to_subtotal"] = np.where(
            df["subtotal_price"].notna() & (df["subtotal_price"] != 0),
            energy_grant / df["subtotal_price"],
            np.nan,
        )

    if "estimated_gross_profit" not in df.columns:
        df["estimated_gross_profit"] = df["subtotal_price"] * df["gross_margin_rate"]

    if "estimated_cost" not in df.columns:
        df["estimated_cost"] = df["subtotal_price"] - df["estimated_gross_profit"]

    return df


def clean_for_modeling(
    df: pd.DataFrame,
    min_product_count: int = MIN_PRODUCT_COUNT,
) -> pd.DataFrame:
    """
    Prepare data using the same logic as the previous CV scripts:
    - keep highest price for quote_id + product duplicate price variations
    - create success label where 1 = success and 0 = fail
    - group rare products as Other
    """
    df_model = df.copy()

    required_basic = {"quote_id", "product", "unit_price", "convert_to_order"}
    missing_basic = required_basic - set(df_model.columns)
    if missing_basic:
        raise ValueError(f"Missing required columns: {sorted(missing_basic)}")

    price_variation = (
        df_model.groupby(["quote_id", "product"])["unit_price"]
        .transform("nunique")
    )

    if "is_highest_price" in df_model.columns:
        before = len(df_model)
        df_model = df_model[
            (price_variation == 1) |
            ((price_variation > 1) & (df_model["is_highest_price"] == 1))
        ].copy()
        print(f"Duplicate price cleaning: {before} -> {len(df_model)} rows")
    else:
        print("Column 'is_highest_price' not found. Duplicate price cleaning skipped.")

    # Target coding based on project definition:
    # convert_to_order = 0 means Success, 1 means Fail
    df_model = df_model[df_model["convert_to_order"].isin([0, 1])].copy()
    df_model[TARGET] = (df_model["convert_to_order"] == 0).astype(int)

    # Rare product grouping
    product_counts = df_model["product"].value_counts()
    rare_products = product_counts[product_counts < min_product_count].index
    df_model["product_model"] = df_model["product"].replace(rare_products, "Other")

    # Make sure every feature exists.
    for col in FEATURES:
        if col not in df_model.columns:
            df_model[col] = np.nan
            print(f"Warning: missing feature '{col}' was created as NaN.")

    # Ensure numeric features are numeric.
    numeric_features = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]
    for col in numeric_features:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    # Ensure categorical feature is string.
    for col in CATEGORICAL_FEATURES:
        df_model[col] = df_model[col].astype("string").fillna("Missing")

    return df_model.reset_index(drop=True)


# ============================================================
# Modeling utilities
# ============================================================

def make_stratify_label(df_model: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.Series:
    """
    Stratify by product_model + success when possible.
    If a product-success group has fewer than n_splits rows,
    fallback to success only for those rare groups.
    """
    label = df_model["product_model"].astype(str) + "_" + df_model[TARGET].astype(str)
    counts = label.value_counts()
    rare_labels = counts[counts < n_splits].index

    safe_label = label.where(~label.isin(rare_labels), df_model[TARGET].astype(str))

    # Final safety check.
    final_counts = safe_label.value_counts()
    if (final_counts < n_splits).any():
        print(
            "Warning: some stratification groups are still smaller than n_splits. "
            "Falling back to target-only stratification."
        )
        safe_label = df_model[TARGET].astype(str)

    return safe_label


def build_hgb_pipeline(params: Dict | None = None) -> Pipeline:
    """Build HistGradientBoosting pipeline."""
    numeric_features = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    default_params = dict(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=RANDOM_STATE,
    )

    if params:
        default_params.update(params)

    model = HistGradientBoostingClassifier(**default_params)

    return Pipeline(steps=[("preprocess", preprocess), ("classifier", model)])


def get_param_distributions() -> Dict[str, List]:
    """
    Reasonable search space for HistGradientBoostingClassifier.
    Kept moderate so it can run on a normal laptop.
    """
    return {
        "classifier__max_iter": [150, 200, 300, 400, 500],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.10],
        "classifier__max_leaf_nodes": [15, 31, 45, 63],
        "classifier__max_depth": [None, 3, 5, 7, 10],
        "classifier__min_samples_leaf": [10, 20, 30, 50],
        "classifier__l2_regularization": [0.0, 0.01, 0.1, 1.0, 5.0],
        "classifier__max_bins": [64, 128, 255],
        "classifier__early_stopping": [True, False],
    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics for one fold."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_prob),
    }

    # These require both classes in the test fold.
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        metrics["log_loss"] = log_loss(y_true, y_prob, labels=[0, 1])
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["log_loss"] = np.nan

    return metrics


def summarize_cv_results(results: pd.DataFrame) -> pd.DataFrame:
    """Create mean and standard deviation summary."""
    metric_cols = [col for col in results.columns if col not in ["model", "fold", "tn", "fp", "fn", "tp"]]
    summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [results[col].mean() for col in metric_cols],
        "std": [results[col].std() for col in metric_cols],
    })
    return summary


def evaluate_pipeline_cv(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: List,
    model_name: str = "Tuned HistGradientBoostingClassifier",
) -> CVOutput:
    """Evaluate one pipeline using precomputed CV splits."""
    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)
        y_prob = fold_model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        row = {
            "model": model_name,
            "fold": fold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        row.update(compute_metrics(y_test, y_pred, y_prob))
        rows.append(row)

        print(
            f"{model_name} fold {fold}: "
            f"ROC AUC={row['roc_auc']:.4f}, "
            f"PR AUC={row['pr_auc']:.4f}, "
            f"F1={row['f1_score']:.4f}, "
            f"Brier={row['brier_score']:.4f}"
        )

    results = pd.DataFrame(rows)
    summary = summarize_cv_results(results)
    return CVOutput(fold_results=results, summary=summary)


def save_final_predictions(df_model: pd.DataFrame, X: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Generate final predicted win probability using model trained on full data."""
    output_cols = [
        "quote_id",
        "product",
        "product_model",
        "gross_margin_rate",
        "unit_price",
        "qty",
        "subtotal_price",
        "convert_to_order",
        TARGET,
    ]

    pred_df = df_model[[col for col in output_cols if col in df_model.columns]].copy()
    pred_df["tuned_predicted_win_probability"] = model.predict_proba(X)[:, 1]
    pred_df["tuned_predicted_win_probability_pct"] = (
        pred_df["tuned_predicted_win_probability"] * 100
    ).round(2)

    return pred_df


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for HistGradientBoosting win probability model.")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/df_preprocessed.csv",
        help="Path to df_preprocessed.csv. Example: dataset/df_preprocessed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Folder for output files. Default: output",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of CV folds. Default: 5",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="Number of random search combinations. Default: 30",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="pr_auc",
        choices=list(SCORING_MAP.keys()),
        help=(
            "Main metric for selecting best parameters. "
            "Recommended: pr_auc for imbalanced win/loss data."
        ),
    )
    parser.add_argument(
        "--min-product-count",
        type=int,
        default=MIN_PRODUCT_COUNT,
        help="Products with fewer rows than this are grouped as Other. Default: 15",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random state. Default: 42",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(args.data)
    print(f"Loaded shape: {df.shape}")

    print("Adding/checking feature engineering columns...")
    df = add_missing_feature_engineering(df)

    print("Preparing modeling dataset...")
    df_model = clean_for_modeling(df, min_product_count=args.min_product_count)
    print(f"Modeling shape: {df_model.shape}")
    print("Target distribution:")
    print(df_model[TARGET].value_counts(normalize=True).rename({1: "Success", 0: "Fail"}))

    X = df_model[FEATURES].copy()
    y = df_model[TARGET].copy()

    stratify_label = make_stratify_label(df_model, n_splits=args.n_splits)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    cv_splits = list(cv.split(X, stratify_label))

    base_pipeline = build_hgb_pipeline()
    param_distributions = get_param_distributions()

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "neg_log_loss": "neg_log_loss",
        "neg_brier_score": "neg_brier_score",
    }
    refit_metric = args.scoring

    print("\nRunning RandomizedSearchCV for HistGradientBoostingClassifier...")
    print(f"n_iter: {args.n_iter}")
    print(f"CV folds: {args.n_splits}")
    print(f"Refit/select best metric: {refit_metric}")

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=cv_splits,
        random_state=args.random_state,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(X, y)

    print("\nBest score:")
    print(f"{refit_metric}: {search.best_score_:.6f}")

    print("\nBest parameters:")
    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    # Save full tuning results
    tuning_results = pd.DataFrame(search.cv_results_)
    tuning_results = tuning_results.sort_values(f"rank_test_{refit_metric}")
    tuning_results.to_csv(
        os.path.join(args.output_dir, "hgb_tuning_cv_results.csv"),
        index=False,
    )

    # Save best params in CSV, TXT, and JSON-friendly format
    best_params_df = pd.DataFrame([
        {"parameter": key, "value": str(value)}
        for key, value in search.best_params_.items()
    ])
    best_params_df.to_csv(
        os.path.join(args.output_dir, "hgb_best_params.csv"),
        index=False,
    )

    with open(os.path.join(args.output_dir, "hgb_best_params.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best refit metric: {refit_metric}\n")
        f.write(f"Best CV score: {search.best_score_:.6f}\n\n")
        f.write("Best parameters:\n")
        for key, value in search.best_params_.items():
            f.write(f"{key}: {value}\n")

    with open(os.path.join(args.output_dir, "hgb_best_params.json"), "w", encoding="utf-8") as f:
        json.dump({key: str(value) for key, value in search.best_params_.items()}, f, indent=2)

    # Evaluate the tuned pipeline with the same folds for easy comparison with previous files.
    # Note: For fully unbiased tuning performance, nested CV is more rigorous.
    print("\nEvaluating tuned HGB model using 5-fold CV...")
    tuned_model = search.best_estimator_
    tuned_output = evaluate_pipeline_cv(
        model=tuned_model,
        X=X,
        y=y,
        cv_splits=cv_splits,
        model_name="Tuned HistGradientBoostingClassifier",
    )

    tuned_results_path = os.path.join(args.output_dir, "hgb_tuned_5fold_results.csv")
    tuned_summary_path = os.path.join(args.output_dir, "hgb_tuned_5fold_summary.csv")
    tuned_output.fold_results.to_csv(tuned_results_path, index=False)
    tuned_output.summary.to_csv(tuned_summary_path, index=False)

    print("\nTuned HGB 5-fold summary:")
    print(tuned_output.summary)

    print("\nTraining final tuned HGB model on full dataset...")
    final_model = clone(tuned_model)
    final_model.fit(X, y)

    final_predictions = save_final_predictions(df_model, X, final_model)
    final_predictions.to_csv(
        os.path.join(args.output_dir, "final_tuned_hgb_predictions.csv"),
        index=False,
    )

    joblib.dump(final_model, os.path.join(args.output_dir, "final_tuned_hgb_model.joblib"))

    print("\nDone. Output files saved in:", os.path.abspath(args.output_dir))
    print("\nMain files to check:")
    print("- hgb_best_params.txt")
    print("- hgb_tuned_5fold_summary.csv")
    print("- final_tuned_hgb_predictions.csv")


if __name__ == "__main__":
    main()
