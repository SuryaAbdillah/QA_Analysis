"""
5-Fold Cross Validation for Simple Neural Network Win Probability Benchmark
=========================================================================

Purpose:
- Use a simple feed-forward neural network as an additional benchmark model
  for quote win probability prediction.
- Evaluate the model using stratified 5-fold cross-validation.
- Train a final neural network model on the full dataset after CV evaluation.
- Save fold results, CV summary, final predictions, and trained model.

Important note:
- This script uses scikit-learn's MLPClassifier, not TensorFlow/PyTorch.
- It is intended as a simple benchmark, not necessarily the main model.
- For tabular business data, tree-based models such as HistGradientBoosting
  are often easier to justify and interpret.

How to run:
    python surya_5fold_neural_network.py --data dataset/df_preprocessed.csv

Alternative path:
    python surya_5fold_neural_network.py --data ../dataset/df_preprocessed.csv

Outputs:
    output/5fold_neural_network_results.csv
    output/5fold_neural_network_summary.csv
    output/final_neural_network_predictions.csv
    output/final_neural_network_model.joblib
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Configuration
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
            "Please check the path. Example: --data dataset/df_preprocessed.csv"
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


def clean_for_modeling(df: pd.DataFrame, min_product_count: int = MIN_PRODUCT_COUNT) -> pd.DataFrame:
    """
    Prepare data using the same logic as the other model scripts:
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
# Model and cross-validation utilities
# ============================================================

def parse_hidden_layers(text: str) -> Tuple[int, ...]:
    """Parse hidden layer size string, e.g. '32,16' -> (32, 16)."""
    try:
        values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Hidden layers must be comma-separated integers, e.g. 32,16") from exc

    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Hidden layers must contain positive integers, e.g. 32,16")

    return values


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with both older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_neural_network_pipeline(
    hidden_layer_sizes: Tuple[int, ...] = (32, 16),
    alpha: float = 0.001,
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    early_stopping: bool = True,
) -> Pipeline:
    """
    Build a simple MLPClassifier pipeline.

    Neural networks are sensitive to feature scale, so StandardScaler is used
    for numeric features after median imputation. Product is one-hot encoded.
    """
    numeric_features = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]

    numeric_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", make_one_hot_encoder()),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", categorical_preprocess, CATEGORICAL_FEATURES),
            ("num", numeric_preprocess, numeric_features),
        ],
        remainder="drop",
    )

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=RANDOM_STATE,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("classifier", model)])


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
    metric_cols = [
        col for col in results.columns
        if col not in ["model", "fold", "tn", "fp", "fn", "tp"]
    ]

    summary = pd.DataFrame({
        "metric": metric_cols,
        "mean": [results[col].mean() for col in metric_cols],
        "std": [results[col].std() for col in metric_cols],
    })

    return summary


def evaluate_neural_network_cv(
    X: pd.DataFrame,
    y: pd.Series,
    stratify_label: pd.Series,
    n_splits: int = N_SPLITS,
    hidden_layer_sizes: Tuple[int, ...] = (32, 16),
    alpha: float = 0.001,
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    early_stopping: bool = True,
) -> CVOutput:
    """Run stratified 5-fold CV for the simple neural network."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    base_model = build_neural_network_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
    )

    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, stratify_label), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(base_model)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        row = {
            "model": "SimpleNeuralNetwork_MLPClassifier",
            "fold": fold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
        row.update(compute_metrics(y_test, y_pred, y_prob))
        rows.append(row)

        print(
            f"Neural Network fold {fold}: "
            f"ROC AUC={row['roc_auc']:.4f}, "
            f"PR AUC={row['pr_auc']:.4f}, "
            f"F1={row['f1_score']:.4f}, "
            f"Brier={row['brier_score']:.4f}"
        )

    results = pd.DataFrame(rows)
    summary = summarize_cv_results(results)
    return CVOutput(fold_results=results, summary=summary)


# ============================================================
# Final model training for full-data prediction
# ============================================================

def train_final_neural_network_model(
    X: pd.DataFrame,
    y: pd.Series,
    hidden_layer_sizes: Tuple[int, ...] = (32, 16),
    alpha: float = 0.001,
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    early_stopping: bool = True,
) -> Pipeline:
    """Train final neural network model on all available data after CV evaluation."""
    model = build_neural_network_pipeline(
        hidden_layer_sizes=hidden_layer_sizes,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
    )
    model.fit(X, y)
    return model


def save_final_neural_network_predictions(df_model: pd.DataFrame, X: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
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
    pred_df["nn_predicted_win_probability"] = model.predict_proba(X)[:, 1]
    pred_df["nn_predicted_win_probability_pct"] = (
        pred_df["nn_predicted_win_probability"] * 100
    ).round(2)

    return pred_df


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV for simple neural network win probability benchmark.")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/df_preprocessed.csv",
        help="Path to df_preprocessed.csv. Example: dataset/df_preprocessed.csv",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=N_SPLITS,
        help="Number of CV folds. Default: 5",
    )
    parser.add_argument(
        "--min-product-count",
        type=int,
        default=MIN_PRODUCT_COUNT,
        help="Products with fewer rows than this are grouped as Other. Default: 15",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers,
        default=(32, 16),
        help="Hidden layer sizes, comma-separated. Default: 32,16",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="L2 regularization strength for MLPClassifier. Default: 0.001",
    )
    parser.add_argument(
        "--learning-rate-init",
        type=float,
        default=0.001,
        help="Initial learning rate for Adam optimizer. Default: 0.001",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum training iterations. Default: 1000",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping. Default: early stopping is enabled.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    early_stopping = not args.no_early_stopping

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

    print("\nRunning 5-fold CV for Simple Neural Network...")
    nn_output = evaluate_neural_network_cv(
        X=X,
        y=y,
        stratify_label=stratify_label,
        n_splits=args.n_splits,
        hidden_layer_sizes=args.hidden_layers,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=early_stopping,
    )

    results_path = os.path.join(OUTPUT_DIR, "5fold_neural_network_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "5fold_neural_network_summary.csv")
    nn_output.fold_results.to_csv(results_path, index=False)
    nn_output.summary.to_csv(summary_path, index=False)

    print("\nSimple Neural Network 5-fold summary:")
    print(nn_output.summary)

    print("\nTraining final neural network model on full dataset for predicted probabilities...")
    final_model = train_final_neural_network_model(
        X=X,
        y=y,
        hidden_layer_sizes=args.hidden_layers,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=early_stopping,
    )
    final_predictions = save_final_neural_network_predictions(df_model, X, final_model)

    final_predictions.to_csv(
        os.path.join(OUTPUT_DIR, "final_neural_network_predictions.csv"),
        index=False,
    )
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, "final_neural_network_model.joblib"))

    print("\nDone. Output files saved in:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
