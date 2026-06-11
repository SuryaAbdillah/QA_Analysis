"""
GMR Recommender Pipeline + Saved Model Bundle + SHAP Explanation
================================================================

Purpose
-------
This script trains the selected HistGradientBoosting win-probability model,
saves a website-callable joblib bundle, recommends Gross Margin Rate (GMR)
values from a user-defined grid, and optionally creates SHAP explanations.

Main outputs
------------
output/final_gmr_recommender_model.joblib
    A fitted sklearn Pipeline. Use this if your website already creates the
    required feature columns.

output/final_gmr_recommender_bundle.joblib
    Recommended for website usage. Contains:
    - fitted sklearn Pipeline
    - feature list
    - categorical/numeric feature list
    - model configuration
    - default GMR grid used during training run
    - expected target coding

output/gmr_recommendations.csv
    Current vs recommended GMR per quote/product row.

output/gmr_grid_summary_by_gmr.csv
    Average predicted win probability and expected profit for each GMR candidate.

Optional SHAP outputs when --make-shap is used:
output/shap_global_feature_importance.csv
output/shap_global_feature_importance.png
output/shap_local_values_sample.csv
output/shap_base_values_sample.csv

How to run
----------
From project root, for example /QA_Analysis:

    python src/surya_gmr_recommender_pipeline_with_shap.py \
        --data dataset/df_preprocessed.csv \
        --gmr-min 0 \
        --gmr-max 0.60 \
        --gmr-interval 0.05 \
        --make-shap

You can also input GMR as percentages:

    python src/surya_gmr_recommender_pipeline_with_shap.py \
        --data dataset/df_preprocessed.csv \
        --gmr-min 0 \
        --gmr-max 60 \
        --gmr-interval 5 \
        --make-shap

Website usage example
---------------------

    import joblib
    import pandas as pd

    bundle = joblib.load("output/final_gmr_recommender_bundle.joblib")
    model = bundle["model"]
    features = bundle["features"]

    input_df = pd.read_csv("some_prepared_quote_features.csv")
    win_prob = model.predict_proba(input_df[features])[:, 1]

Important note
--------------
The saved sklearn Pipeline expects the same feature columns listed in
bundle["features"]. For raw website input, apply the same feature engineering
logic before calling predict_proba.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
MIN_PRODUCT_COUNT = 15
OUTPUT_DIR = "output"
TARGET = "success"
CATEGORICAL_FEATURES = ["product_model"]

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

NUMERIC_FEATURES = [col for col in FEATURES if col not in CATEGORICAL_FEATURES]

# Initial HGB was selected as final because it performed better overall than tuned HGB
# in previous evaluation, especially in F1, PR-AUC, recall, and Brier Score.
INITIAL_HGB_PARAMS: Dict[str, object] = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.1,
    "random_state": RANDOM_STATE,
}

# Best params from RandomizedSearchCV. Kept as an optional config, not default.
TUNED_HGB_PARAMS: Dict[str, object] = {
    "min_samples_leaf": 10,
    "max_leaf_nodes": 63,
    "max_iter": 400,
    "max_depth": None,
    "max_bins": 255,
    "learning_rate": 0.01,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "random_state": RANDOM_STATE,
}


# ============================================================
# Data loading and feature engineering
# ============================================================

def load_data(data_path: str) -> pd.DataFrame:
    """Load quotation data."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Example from project root: --data dataset/df_preprocessed.csv"
        )

    df = pd.read_csv(data_path)

    if "convert_to_order" not in df.columns and "Success" in df.columns:
        df = df.rename(columns={"Success": "convert_to_order"})

    if "convert_to_order" not in df.columns:
        raise ValueError(
            "Column 'convert_to_order' is required. Expected coding: 0 = Success, 1 = Fail."
        )

    return df


def add_missing_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add required helper features if they are missing."""
    df = df.copy()

    required_base_cols = ["unit_price", "subtotal_price", "gross_margin_rate", "qty"]
    for col in required_base_cols:
        if col not in df.columns:
            raise ValueError(f"Required column is missing: {col}")

    # Competitor raw columns fallback
    for col in ["competitor_a", "competitor_b", "competitor_c"]:
        if col not in df.columns:
            df[col] = np.nan

    # Competitor availability flags
    for source_col, flag_col in [
        ("competitor_a", "is_compe_a"),
        ("competitor_b", "is_compe_b"),
        ("competitor_c", "is_compe_c"),
    ]:
        if flag_col not in df.columns:
            df[flag_col] = df[source_col].notna().astype(int)

    if "competitor_count_available" not in df.columns:
        df["competitor_count_available"] = df[["competitor_a", "competitor_b", "competitor_c"]].notna().sum(axis=1)

    if "known_num_compe" not in df.columns:
        df["known_num_compe"] = df["competitor_count_available"]

    # Competitor summary prices
    if "avg_competitor_price" not in df.columns:
        df["avg_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].mean(axis=1)

    if "min_competitor_price" not in df.columns:
        df["min_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].min(axis=1)

    if "max_competitor_price" not in df.columns:
        df["max_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].max(axis=1)

    # Price order within quote_id + product
    if "price_order" not in df.columns:
        if {"quote_id", "product", "unit_price"}.issubset(df.columns):
            df["price_order"] = (
                df.groupby(["quote_id", "product"])["unit_price"]
                .rank(method="dense", ascending=False)
                .astype(int)
            )
        else:
            df["price_order"] = 1

    if "is_highest_price" not in df.columns:
        df["is_highest_price"] = (df["price_order"] == 1).astype(int)

    # Grant and profit
    if "energy_grant_amount" not in df.columns:
        df["energy_grant_amount"] = 0

    df["energy_grant_amount"] = pd.to_numeric(df["energy_grant_amount"], errors="coerce").fillna(0)

    if "estimated_gross_profit" not in df.columns:
        df["estimated_gross_profit"] = df["subtotal_price"] * df["gross_margin_rate"]

    if "estimated_cost" not in df.columns:
        df["estimated_cost"] = df["subtotal_price"] - df["estimated_gross_profit"]

    if "effective_price_after_grant" not in df.columns:
        df["effective_price_after_grant"] = df["subtotal_price"] - df["energy_grant_amount"]

    if "grant_ratio_to_subtotal" not in df.columns:
        df["grant_ratio_to_subtotal"] = np.where(
            df["subtotal_price"].notna() & (df["subtotal_price"] != 0),
            df["energy_grant_amount"] / df["subtotal_price"],
            np.nan,
        )

    # Competitor gap features
    df = recalculate_price_related_features(df)
    return df


def recalculate_price_related_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate price features affected by unit_price/subtotal/GMR changes."""
    df = df.copy()

    df["price_gap_avg_competitor"] = df["unit_price"] - df["avg_competitor_price"]
    df["price_gap_avg_competitor_pct"] = np.where(
        df["avg_competitor_price"].notna() & (df["avg_competitor_price"] != 0),
        (df["price_gap_avg_competitor"] / df["avg_competitor_price"]) * 100,
        np.nan,
    )

    df["price_gap_min_competitor"] = df["unit_price"] - df["min_competitor_price"]
    df["price_gap_min_competitor_pct"] = np.where(
        df["min_competitor_price"].notna() & (df["min_competitor_price"] != 0),
        (df["price_gap_min_competitor"] / df["min_competitor_price"]) * 100,
        np.nan,
    )

    df["higher_than_avg_competitor"] = (df["price_gap_avg_competitor"] > 0).astype(int)
    df["is_lower_than_competitor"] = (df["price_gap_min_competitor"] < 0).astype(int)

    df["effective_price_after_grant"] = df["subtotal_price"] - df["energy_grant_amount"].fillna(0)
    df["grant_ratio_to_subtotal"] = np.where(
        df["subtotal_price"].notna() & (df["subtotal_price"] != 0),
        df["energy_grant_amount"].fillna(0) / df["subtotal_price"],
        np.nan,
    )

    df["estimated_gross_profit"] = df["subtotal_price"] * df["gross_margin_rate"]
    df["estimated_cost"] = df["subtotal_price"] - df["estimated_gross_profit"]

    return df


def clean_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare row-level modeling data:
    - keep highest price row when same quote_id + product has price variations
    - remove invalid target rows
    - create success label
    - group rare products into Other
    """
    df_model = df.copy()

    required_basic = {"quote_id", "product", "unit_price", "convert_to_order"}
    missing_basic = required_basic - set(df_model.columns)
    if missing_basic:
        raise ValueError(f"Missing required columns: {sorted(missing_basic)}")

    before = len(df_model)
    price_variation = (
        df_model.groupby(["quote_id", "product"])["unit_price"]
        .transform("nunique")
    )
    df_model = df_model[
        (price_variation == 1) |
        ((price_variation > 1) & (df_model["is_highest_price"] == 1))
    ].copy()
    print(f"Duplicate price cleaning: {before} -> {len(df_model)} rows")

    # Target coding: convert_to_order = 0 means Success, 1 means Fail
    df_model = df_model[df_model["convert_to_order"].isin([0, 1])].copy()
    df_model[TARGET] = (df_model["convert_to_order"] == 0).astype(int)

    # Rare product grouping
    product_counts = df_model["product"].value_counts()
    rare_products = product_counts[product_counts < MIN_PRODUCT_COUNT].index
    df_model["product_model"] = df_model["product"].replace(rare_products, "Other")

    # Make sure features exist and types are safe
    for col in FEATURES:
        if col not in df_model.columns:
            df_model[col] = np.nan
            print(f"Warning: missing feature '{col}' was created as NaN.")

    for col in NUMERIC_FEATURES:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        df_model[col] = df_model[col].astype("string").fillna("Missing")

    return df_model.reset_index(drop=True)


# ============================================================
# Model pipeline
# ============================================================

def build_hgb_pipeline(model_config: str = "initial") -> Pipeline:
    """
    Build HGB pipeline.

    model_config:
        - initial: selected final model based on better overall CV performance
        - tuned: best RandomizedSearchCV parameters, optional comparison
    """
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
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )

    if model_config == "initial":
        params = INITIAL_HGB_PARAMS.copy()
    elif model_config == "tuned":
        params = TUNED_HGB_PARAMS.copy()
    else:
        raise ValueError("model_config must be either 'initial' or 'tuned'.")

    classifier = HistGradientBoostingClassifier(**params)
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)])


def save_model_bundle(
    model: Pipeline,
    output_dir: str,
    model_config: str,
    gmr_grid: np.ndarray,
    min_win_prob: float,
    objective: str,
) -> Tuple[str, str]:
    """Save both the fitted Pipeline and a richer bundle for website use."""
    model_path = os.path.join(output_dir, "final_gmr_recommender_model.joblib")
    bundle_path = os.path.join(output_dir, "final_gmr_recommender_bundle.joblib")

    joblib.dump(model, model_path)

    bundle = {
        "model": model,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "target_definition": "success = 1 when convert_to_order == 0; fail = 0 when convert_to_order == 1",
        "model_name": "HistGradientBoostingClassifier",
        "model_config": model_config,
        "initial_hgb_params": INITIAL_HGB_PARAMS,
        "tuned_hgb_params": TUNED_HGB_PARAMS,
        "selected_hgb_params": INITIAL_HGB_PARAMS if model_config == "initial" else TUNED_HGB_PARAMS,
        "gmr_grid": gmr_grid.tolist(),
        "gmr_grid_pct": (gmr_grid * 100).round(4).tolist(),
        "min_win_prob": min_win_prob,
        "objective": objective,
        "required_raw_columns_note": (
            "For raw website input, create the same engineered columns before prediction. "
            "The model pipeline expects input_df[features]."
        ),
    }
    joblib.dump(bundle, bundle_path)

    # JSON metadata is convenient for non-Python website parts.
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    with open(os.path.join(output_dir, "final_gmr_recommender_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return model_path, bundle_path


# ============================================================
# GMR grid and scenario simulation
# ============================================================

def normalize_gmr_input(value: float) -> float:
    """Accept both decimal and percentage style: 0.30 -> 0.30, 30 -> 0.30."""
    value = float(value)
    if abs(value) > 1:
        value = value / 100.0
    return value


def build_gmr_grid(gmr_min: float, gmr_max: float, gmr_interval: float) -> np.ndarray:
    """Create GMR grid from min, max, and interval."""
    gmr_min = normalize_gmr_input(gmr_min)
    gmr_max = normalize_gmr_input(gmr_max)
    gmr_interval = normalize_gmr_input(gmr_interval)

    if gmr_interval <= 0:
        raise ValueError("gmr_interval must be positive.")
    if gmr_min > gmr_max:
        raise ValueError("gmr_min must be smaller than or equal to gmr_max.")
    if gmr_max >= 0.95:
        raise ValueError("gmr_max must be lower than 0.95 because price = cost / (1 - GMR).")

    grid = np.arange(gmr_min, gmr_max + (gmr_interval / 2), gmr_interval)
    grid = np.round(grid, 10)
    grid = grid[grid < 0.95]

    if len(grid) == 0:
        raise ValueError("GMR grid is empty. Please check min, max, and interval.")

    return grid


def apply_candidate_gmr(base_df: pd.DataFrame, candidate_gmr: float) -> pd.DataFrame:
    """
    Apply one candidate GMR to all rows.

    Assumption:
    - estimated_cost remains fixed
    - new subtotal is calculated from:
          GMR = (Price - Cost) / Price
          Price = Cost / (1 - GMR)
    - unit_price is scaled according to subtotal change
    """
    df = base_df.copy()
    candidate_gmr = float(candidate_gmr)

    original_subtotal = pd.to_numeric(df["subtotal_price"], errors="coerce")
    original_unit_price = pd.to_numeric(df["unit_price"], errors="coerce")
    estimated_cost = pd.to_numeric(df["estimated_cost"], errors="coerce")

    missing_cost = estimated_cost.isna()
    if missing_cost.any():
        estimated_cost = estimated_cost.where(
            ~missing_cost,
            original_subtotal * (1 - pd.to_numeric(df["gross_margin_rate"], errors="coerce")),
        )

    new_subtotal = estimated_cost / (1 - candidate_gmr)

    scale = np.where(
        original_subtotal.notna() & (original_subtotal != 0),
        new_subtotal / original_subtotal,
        np.nan,
    )
    new_unit_price = original_unit_price * scale

    qty_num = pd.to_numeric(df["qty"], errors="coerce")
    fallback_unit_price = np.where(qty_num.notna() & (qty_num != 0), new_subtotal / qty_num, original_unit_price)
    new_unit_price = np.where(pd.isna(new_unit_price), fallback_unit_price, new_unit_price)

    df["candidate_gmr"] = candidate_gmr
    df["gross_margin_rate"] = candidate_gmr
    df["subtotal_price"] = new_subtotal
    df["unit_price"] = new_unit_price
    df["estimated_cost"] = estimated_cost
    df["estimated_gross_profit"] = df["subtotal_price"] * candidate_gmr

    return recalculate_price_related_features(df)


def create_gmr_scenarios(df_model: pd.DataFrame, gmr_grid: Iterable[float]) -> pd.DataFrame:
    """Create candidate rows for every row_id x candidate_gmr."""
    base = df_model.copy().reset_index(drop=True)
    base["row_id"] = np.arange(len(base))
    base["current_gmr"] = base["gross_margin_rate"]
    base["current_unit_price"] = base["unit_price"]
    base["current_subtotal_price"] = base["subtotal_price"]
    base["current_estimated_gross_profit"] = base["estimated_gross_profit"]

    scenario_parts = [apply_candidate_gmr(base, gmr) for gmr in gmr_grid]
    scenarios = pd.concat(scenario_parts, ignore_index=True)

    for col in NUMERIC_FEATURES:
        scenarios[col] = pd.to_numeric(scenarios[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        scenarios[col] = scenarios[col].astype("string").fillna("Missing")

    return scenarios


# ============================================================
# Recommendation logic
# ============================================================

def add_predictions_and_expected_profit(scenarios: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Predict win probability and expected profit for all GMR candidates."""
    scenarios = scenarios.copy()
    X_scenarios = scenarios[FEATURES].copy()

    scenarios["predicted_win_probability"] = model.predict_proba(X_scenarios)[:, 1]
    scenarios["predicted_win_probability_pct"] = (scenarios["predicted_win_probability"] * 100).round(2)
    scenarios["candidate_estimated_gross_profit"] = scenarios["estimated_gross_profit"]
    scenarios["expected_profit"] = scenarios["predicted_win_probability"] * scenarios["candidate_estimated_gross_profit"]

    return scenarios


def select_recommendations(
    scenarios: pd.DataFrame,
    min_win_prob: float = 0.0,
    objective: str = "expected_profit",
) -> pd.DataFrame:
    """Select best GMR candidate per original row."""
    if objective not in {"expected_profit", "predicted_win_probability", "candidate_estimated_gross_profit"}:
        raise ValueError("objective must be one of: expected_profit, predicted_win_probability, candidate_estimated_gross_profit")

    rows = []
    for row_id, group in scenarios.groupby("row_id", sort=False):
        eligible = group[group["predicted_win_probability"] >= min_win_prob]
        used_fallback = False

        if len(eligible) == 0:
            pool = group
            used_fallback = True
        else:
            pool = eligible

        best_idx = pool[objective].idxmax()
        best_row = scenarios.loc[best_idx].copy()
        best_row["recommendation_objective"] = objective
        best_row["min_win_prob_constraint"] = min_win_prob
        best_row["used_fallback_because_no_candidate_met_min_win_prob"] = used_fallback
        rows.append(best_row)

    return pd.DataFrame(rows).reset_index(drop=True)


def build_current_prediction_table(df_model: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """Generate predictions for current/original pricing."""
    current = df_model.copy().reset_index(drop=True)
    current["row_id"] = np.arange(len(current))
    X_current = current[FEATURES].copy()

    current["current_predicted_win_probability"] = model.predict_proba(X_current)[:, 1]
    current["current_predicted_win_probability_pct"] = (current["current_predicted_win_probability"] * 100).round(2)
    current["current_expected_profit"] = current["current_predicted_win_probability"] * current["estimated_gross_profit"]

    keep_cols = [
        "row_id", "quote_id", "product", "product_model", "kw", "qty",
        "unit_price", "subtotal_price", "gross_margin_rate", "energy_grant_amount",
        "estimated_cost", "estimated_gross_profit", "avg_competitor_price",
        "price_gap_avg_competitor_pct", "convert_to_order", TARGET,
        "current_predicted_win_probability", "current_predicted_win_probability_pct",
        "current_expected_profit",
    ]
    keep_cols = [col for col in keep_cols if col in current.columns]
    return current[keep_cols].copy()


def format_recommendation_output(recommendations: pd.DataFrame, current_table: pd.DataFrame) -> pd.DataFrame:
    """Create final recommendation table with current vs recommended comparison."""
    rec_keep = [
        "row_id", "candidate_gmr", "unit_price", "subtotal_price",
        "candidate_estimated_gross_profit", "predicted_win_probability",
        "predicted_win_probability_pct", "expected_profit",
        "price_gap_avg_competitor_pct", "higher_than_avg_competitor",
        "is_lower_than_competitor", "recommendation_objective",
        "min_win_prob_constraint", "used_fallback_because_no_candidate_met_min_win_prob",
    ]
    rec_keep = [col for col in rec_keep if col in recommendations.columns]

    rec = recommendations[rec_keep].copy()
    rec = rec.rename(columns={
        "candidate_gmr": "recommended_gmr",
        "unit_price": "recommended_unit_price",
        "subtotal_price": "recommended_subtotal_price",
        "candidate_estimated_gross_profit": "recommended_estimated_gross_profit",
        "predicted_win_probability": "recommended_predicted_win_probability",
        "predicted_win_probability_pct": "recommended_predicted_win_probability_pct",
        "expected_profit": "recommended_expected_profit",
        "price_gap_avg_competitor_pct": "recommended_price_gap_avg_competitor_pct",
    })

    out = current_table.merge(rec, on="row_id", how="left")

    out["delta_gmr"] = out["recommended_gmr"] - out["gross_margin_rate"]
    out["delta_unit_price"] = out["recommended_unit_price"] - out["unit_price"]
    out["delta_subtotal_price"] = out["recommended_subtotal_price"] - out["subtotal_price"]
    out["delta_predicted_win_probability"] = out["recommended_predicted_win_probability"] - out["current_predicted_win_probability"]
    out["delta_expected_profit"] = out["recommended_expected_profit"] - out["current_expected_profit"]

    out["current_gmr_pct"] = (out["gross_margin_rate"] * 100).round(2)
    out["recommended_gmr_pct"] = (out["recommended_gmr"] * 100).round(2)
    out["delta_gmr_pct_point"] = (out["delta_gmr"] * 100).round(2)
    out["delta_predicted_win_probability_pct_point"] = (out["delta_predicted_win_probability"] * 100).round(2)

    first_cols = [
        "quote_id", "product", "product_model", "kw", "qty",
        "current_gmr_pct", "recommended_gmr_pct", "delta_gmr_pct_point",
        "unit_price", "recommended_unit_price", "delta_unit_price",
        "subtotal_price", "recommended_subtotal_price", "delta_subtotal_price",
        "estimated_cost", "estimated_gross_profit", "recommended_estimated_gross_profit",
        "current_predicted_win_probability_pct", "recommended_predicted_win_probability_pct",
        "delta_predicted_win_probability_pct_point", "current_expected_profit",
        "recommended_expected_profit", "delta_expected_profit",
        "avg_competitor_price", "price_gap_avg_competitor_pct",
        "recommended_price_gap_avg_competitor_pct", "convert_to_order", TARGET,
    ]
    first_cols = [col for col in first_cols if col in out.columns]
    remaining_cols = [col for col in out.columns if col not in first_cols]

    return out[first_cols + remaining_cols].copy()


def summarize_scenarios_by_gmr(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Summarize average outcome by candidate GMR."""
    summary = (
        scenarios
        .groupby("candidate_gmr", observed=True)
        .agg(
            total_rows=("row_id", "count"),
            avg_predicted_win_probability=("predicted_win_probability", "mean"),
            median_predicted_win_probability=("predicted_win_probability", "median"),
            avg_candidate_estimated_gross_profit=("candidate_estimated_gross_profit", "mean"),
            avg_expected_profit=("expected_profit", "mean"),
            median_expected_profit=("expected_profit", "median"),
            avg_price_gap_avg_competitor_pct=("price_gap_avg_competitor_pct", "mean"),
        )
        .reset_index()
    )
    summary["candidate_gmr_pct"] = (summary["candidate_gmr"] * 100).round(2)
    summary["avg_predicted_win_probability_pct"] = (summary["avg_predicted_win_probability"] * 100).round(2)
    return summary


# ============================================================
# SHAP explanation
# ============================================================

def get_transformed_feature_names(model: Pipeline) -> List[str]:
    """Get feature names after preprocessing."""
    preprocess = model.named_steps["preprocess"]
    try:
        names = preprocess.get_feature_names_out().tolist()
    except Exception:
        names = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    # Make names easier to read.
    cleaned = []
    for name in names:
        name = str(name)
        name = name.replace("cat__", "")
        name = name.replace("num__", "")
        cleaned.append(name)
    return cleaned


def make_shap_outputs(
    model: Pipeline,
    X: pd.DataFrame,
    output_dir: str,
    background_size: int = 300,
    sample_size: int = 1000,
    random_state: int = RANDOM_STATE,
) -> None:
    """
    Create SHAP explanation outputs for the fitted HGB pipeline.

    Note:
    SHAP is calculated on the transformed feature matrix, because the model is
    inside a sklearn Pipeline. The saved feature names are cleaned to match the
    original feature names as much as possible.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Please run: pip install shap"
        ) from exc

    os.makedirs(output_dir, exist_ok=True)

    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]

    rng = np.random.default_rng(random_state)
    n_rows = len(X)
    background_n = min(background_size, n_rows)
    sample_n = min(sample_size, n_rows)

    background_idx = rng.choice(n_rows, size=background_n, replace=False)
    sample_idx = rng.choice(n_rows, size=sample_n, replace=False)

    X_background_raw = X.iloc[background_idx].copy()
    X_sample_raw = X.iloc[sample_idx].copy()

    X_background = preprocess.transform(X_background_raw)
    X_sample = preprocess.transform(X_sample_raw)

    if hasattr(X_background, "toarray"):
        X_background = X_background.toarray()
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    feature_names = get_transformed_feature_names(model)

    print("Creating SHAP explainer...")
    # shap.Explainer will choose the best available explainer. For tree models,
    # it usually uses a tree-based explainer; otherwise it falls back gracefully.
    explainer = shap.Explainer(classifier, X_background, feature_names=feature_names)

    print(f"Calculating SHAP values for {sample_n} sampled rows...")
    shap_values = explainer(X_sample)
    values = np.asarray(shap_values.values)

    # Some binary classifiers return shape: (n_samples, n_features, n_outputs).
    # If so, use the positive/success class when available.
    if values.ndim == 3:
        if values.shape[2] > 1:
            values = values[:, :, 1]
        else:
            values = values[:, :, 0]

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    mean_abs = np.abs(values).mean(axis=0)
    importance = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = np.arange(1, len(importance) + 1)

    importance_path = os.path.join(output_dir, "shap_global_feature_importance.csv")
    importance.to_csv(importance_path, index=False)

    # Save local SHAP values for sample rows. This is useful for checking or for
    # building simple explanation tables in a web app.
    local_values = pd.DataFrame(values, columns=feature_names)
    local_values.insert(0, "original_row_index", X.index[sample_idx])
    local_path = os.path.join(output_dir, "shap_local_values_sample.csv")
    local_values.to_csv(local_path, index=False)

    # Save base values separately.
    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim > 1:
        base_values = base_values[:, -1]
    base_df = pd.DataFrame({
        "original_row_index": X.index[sample_idx],
        "base_value": base_values,
    })
    base_path = os.path.join(output_dir, "shap_base_values_sample.csv")
    base_df.to_csv(base_path, index=False)

    # Simple global importance plot without relying on interactive JS.
    top_n = min(20, len(importance))
    plot_df = importance.head(top_n).sort_values("mean_abs_shap", ascending=True)

    plt.figure(figsize=(10, max(5, top_n * 0.35)))
    plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "shap_global_feature_importance.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Try saving explainer. This is convenient, but not required for deployment.
    # If joblib fails, the CSV outputs are still available.
    explainer_path = os.path.join(output_dir, "shap_explainer.joblib")
    try:
        joblib.dump(explainer, explainer_path)
        print("Saved SHAP explainer:", explainer_path)
    except Exception as exc:
        print("Warning: could not save SHAP explainer object. CSV/PNG outputs were still saved.")
        print("Reason:", str(exc))

    print("SHAP outputs saved:")
    print("-", importance_path)
    print("-", plot_path)
    print("-", local_path)
    print("-", base_path)


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="GMR recommender using HistGradientBoosting win-probability model with optional SHAP.")
    parser.add_argument("--data", type=str, default="dataset/df_preprocessed.csv", help="Path to df_preprocessed.csv.")
    parser.add_argument("--gmr-min", type=float, required=True, help="Minimum GMR. Accepts decimal 0.20 or percent 20.")
    parser.add_argument("--gmr-max", type=float, required=True, help="Maximum GMR. Accepts decimal 0.60 or percent 60.")
    parser.add_argument("--gmr-interval", type=float, required=True, help="GMR interval. Accepts decimal 0.05 or percent 5.")
    parser.add_argument("--model-config", type=str, choices=["initial", "tuned"], default="initial", help="Default: initial.")
    parser.add_argument("--min-win-prob", type=float, default=0.0, help="Minimum predicted win probability constraint. Default: 0.0.")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["expected_profit", "predicted_win_probability", "candidate_estimated_gross_profit"],
        default="expected_profit",
        help="Objective used to choose recommended GMR. Default: expected_profit.",
    )
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory. Default: output.")
    parser.add_argument("--save-all-scenarios", action="store_true", help="Save all row_id x GMR scenarios.")
    parser.add_argument("--make-shap", action="store_true", help="Create SHAP global/local explanation outputs.")
    parser.add_argument("--shap-background-size", type=int, default=300, help="Rows used as SHAP background. Default: 300.")
    parser.add_argument("--shap-sample-size", type=int, default=1000, help="Rows explained by SHAP. Default: 1000.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(args.data)
    print(f"Loaded shape: {df.shape}")

    print("Adding/checking feature engineering columns...")
    df = add_missing_feature_engineering(df)

    print("Preparing modeling dataset...")
    df_model = clean_for_modeling(df)
    print(f"Modeling shape: {df_model.shape}")
    print("Target distribution:")
    print(df_model[TARGET].value_counts(normalize=True).rename({1: "Success", 0: "Fail"}))

    gmr_grid = build_gmr_grid(args.gmr_min, args.gmr_max, args.gmr_interval)
    print("GMR grid:", ", ".join([f"{g * 100:.2f}%" for g in gmr_grid]))

    print(f"Training final HGB model using model_config='{args.model_config}'...")
    X = df_model[FEATURES].copy()
    y = df_model[TARGET].copy()
    model = build_hgb_pipeline(model_config=args.model_config)
    model.fit(X, y)

    model_path, bundle_path = save_model_bundle(
        model=model,
        output_dir=args.output_dir,
        model_config=args.model_config,
        gmr_grid=gmr_grid,
        min_win_prob=args.min_win_prob,
        objective=args.objective,
    )

    print("Creating GMR candidate scenarios...")
    scenarios = create_gmr_scenarios(df_model, gmr_grid)

    print("Predicting win probability for every GMR candidate...")
    scenarios = add_predictions_and_expected_profit(scenarios, model)

    print("Selecting recommended GMR per row...")
    recommendations = select_recommendations(
        scenarios,
        min_win_prob=args.min_win_prob,
        objective=args.objective,
    )

    current_table = build_current_prediction_table(df_model, model)
    final_recommendations = format_recommendation_output(recommendations, current_table)
    gmr_summary = summarize_scenarios_by_gmr(scenarios)

    rec_path = os.path.join(args.output_dir, "gmr_recommendations.csv")
    summary_path = os.path.join(args.output_dir, "gmr_grid_summary_by_gmr.csv")
    scenarios_path = os.path.join(args.output_dir, "gmr_grid_scenarios.csv")

    final_recommendations.to_csv(rec_path, index=False)
    gmr_summary.to_csv(summary_path, index=False)

    if args.save_all_scenarios:
        scenarios.to_csv(scenarios_path, index=False)

    if args.make_shap:
        make_shap_outputs(
            model=model,
            X=X,
            output_dir=args.output_dir,
            background_size=args.shap_background_size,
            sample_size=args.shap_sample_size,
            random_state=RANDOM_STATE,
        )

    print("\nDone. Output files saved in:", os.path.abspath(args.output_dir))
    print("Main files:")
    print("-", rec_path)
    print("-", summary_path)
    print("-", model_path)
    print("-", bundle_path)
    print("-", os.path.join(args.output_dir, "final_gmr_recommender_metadata.json"))
    if args.save_all_scenarios:
        print("-", scenarios_path)

    print("\nTop 10 rows by delta_expected_profit:")
    preview_cols = [
        "quote_id", "product", "current_gmr_pct", "recommended_gmr_pct",
        "current_predicted_win_probability_pct", "recommended_predicted_win_probability_pct",
        "current_expected_profit", "recommended_expected_profit", "delta_expected_profit",
    ]
    preview_cols = [col for col in preview_cols if col in final_recommendations.columns]
    print(
        final_recommendations
        .sort_values("delta_expected_profit", ascending=False)
        [preview_cols]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
