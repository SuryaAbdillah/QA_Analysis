"""
Realistic GMR Threshold Return Lift Analysis
===========================================

Purpose
-------
Calculate portfolio/row-level potential expected return lift using an already-trained
GMR recommender model bundle. This version simulates candidate GMR values more
realistically by updating all price-related and margin-related engineered features.

This script does NOT retrain the model. It loads a saved bundle such as:
    output/final_gmr_recommender_bundle.joblib

Main business logic
-------------------
For every quote/product row:
1. Predict current win probability using the saved model.
2. Calculate current expected return:
       current_expected_return = current_win_probability * current_estimated_gross_profit
3. Generate candidate GMR values from a grid.
4. For each candidate GMR, keep estimated cost fixed and recompute:
       new_subtotal_price = estimated_cost / (1 - candidate_gmr)
       new_unit_price = new_subtotal_price / qty
       new_estimated_gross_profit = new_subtotal_price - estimated_cost
   Then recompute competitor gap, grant ratio, effective price after grant, etc.
5. Predict win probability for every candidate.
6. For each minimum win probability threshold, e.g. 60%, 75%, 90%:
       filter candidates with predicted_win_probability >= threshold
       select the candidate with the highest expected return per row
7. Compare recommended expected return against current expected return.

Recommended run from project root, e.g. /QA_Analysis:
-----------------------------------------------------
python src/gmr_threshold_return_lift_realistic.py \
    --data dataset/df_preprocessed.csv \
    --bundle output/final_gmr_recommender_bundle.joblib \
    --gmr-min 0 \
    --gmr-max 0.60 \
    --gmr-interval 0.05 \
    --thresholds 0.60,0.75,0.90

Recommended conservative run with guardrails:
---------------------------------------------
python src/gmr_threshold_return_lift_realistic.py \
    --data dataset/df_preprocessed.csv \
    --bundle output/final_gmr_recommender_bundle.joblib \
    --gmr-min 0 \
    --gmr-max 0.60 \
    --gmr-interval 0.05 \
    --thresholds 0.60,0.75,0.90 \
    --max-gmr-change 0.15 \
    --max-price-increase-pct 20 \
    --max-price-gap-min-pct 25

Notes
-----
- Use the wording "potential expected return lift", not "actual profit increase".
- This is still a model-based simulation. It should be validated with business rules.
- GMR inputs can be decimals or percentages. Example: 0.60 and 60 are both accepted.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Dict, Iterable, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
MIN_PRODUCT_COUNT = 15
DEFAULT_OUTPUT_DIR = "output"


# ============================================================
# Basic utilities
# ============================================================

def normalize_decimal_or_pct(value: Optional[float]) -> Optional[float]:
    """Accept both decimal and percentage values.

    Examples
    --------
    0.60 -> 0.60
    60   -> 0.60
    5    -> 0.05
    """
    if value is None:
        return None
    value = float(value)
    if abs(value) > 1:
        value = value / 100.0
    return value


def parse_float_list(text: str) -> List[float]:
    """Parse comma-separated decimals/percentages into decimals."""
    return [normalize_decimal_or_pct(float(x.strip())) for x in text.split(",") if x.strip()]


def build_gmr_grid(gmr_min: float, gmr_max: float, gmr_interval: float) -> np.ndarray:
    """Create candidate GMR grid from min, max, and interval."""
    gmr_min = normalize_decimal_or_pct(gmr_min)
    gmr_max = normalize_decimal_or_pct(gmr_max)
    gmr_interval = normalize_decimal_or_pct(gmr_interval)

    if gmr_interval is None or gmr_interval <= 0:
        raise ValueError("gmr_interval must be positive.")
    if gmr_min is None or gmr_max is None:
        raise ValueError("gmr_min and gmr_max must not be None.")
    if gmr_min > gmr_max:
        raise ValueError("gmr_min must be smaller than or equal to gmr_max.")
    if gmr_max >= 0.95:
        raise ValueError("gmr_max must be lower than 0.95 because price = cost / (1 - GMR).")

    grid = np.arange(gmr_min, gmr_max + (gmr_interval / 2), gmr_interval)
    grid = np.round(grid, 10)
    grid = grid[grid < 0.95]

    if len(grid) == 0:
        raise ValueError("GMR grid is empty. Please check --gmr-min, --gmr-max, and --gmr-interval.")

    return grid


def safe_pct(numerator: float, denominator: float) -> float:
    """Return percentage numerator / denominator * 100; NaN if invalid denominator."""
    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return (numerator / denominator) * 100


def threshold_label(threshold: float) -> str:
    return str(int(round(threshold * 100)))


# ============================================================
# Load saved model bundle and data
# ============================================================

def load_bundle(bundle_path: str) -> Dict:
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f"Model bundle not found: {bundle_path}\n"
            "Expected example: --bundle output/final_gmr_recommender_bundle.joblib"
        )

    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("The loaded joblib must be a dictionary bundle.")
    if "model" not in bundle or "features" not in bundle:
        raise ValueError("Bundle must contain at least keys: 'model' and 'features'.")

    return bundle


def load_data(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Example from project root: --data dataset/df_preprocessed.csv"
        )

    df = pd.read_csv(data_path)
    if "convert_to_order" not in df.columns and "Success" in df.columns:
        df = df.rename(columns={"Success": "convert_to_order"})

    if "convert_to_order" not in df.columns:
        raise ValueError("Column 'convert_to_order' is required. Expected coding: 0 = Success, 1 = Fail.")

    return df


# ============================================================
# Feature engineering consistent with training data
# ============================================================

def to_numeric_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def recalculate_features_after_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate engineered features affected by unit_price/subtotal/GMR.

    This is the key function that makes the simulation more realistic.
    If candidate GMR changes, price and related variables also change.
    """
    df = df.copy()

    # Make sure numeric columns are numeric.
    numeric_cols = [
        "unit_price", "qty", "subtotal_price", "gross_margin_rate", "energy_grant_amount",
        "competitor_a", "competitor_b", "competitor_c", "avg_competitor_price",
        "min_competitor_price", "max_competitor_price", "estimated_cost",
    ]
    df = to_numeric_if_exists(df, numeric_cols)

    if "energy_grant_amount" not in df.columns:
        df["energy_grant_amount"] = 0.0
    df["energy_grant_amount"] = df["energy_grant_amount"].fillna(0.0)

    # Estimated gross profit and cost. In scenario simulation, estimated_cost is kept fixed.
    df["estimated_gross_profit"] = df["subtotal_price"] - df["estimated_cost"]

    # Effective customer price after grant.
    df["effective_price_after_grant"] = df["subtotal_price"] - df["energy_grant_amount"]
    df["grant_ratio_to_subtotal"] = np.where(
        df["subtotal_price"].notna() & (df["subtotal_price"] != 0),
        df["energy_grant_amount"] / df["subtotal_price"],
        np.nan,
    )

    # Competitor price gaps.
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

    df["higher_than_avg_competitor"] = np.where(
        df["avg_competitor_price"].notna(),
        (df["unit_price"] > df["avg_competitor_price"]).astype(int),
        np.nan,
    )
    df["is_lower_than_competitor"] = np.where(
        df["min_competitor_price"].notna(),
        (df["unit_price"] < df["min_competitor_price"]).astype(int),
        np.nan,
    )

    return df


def add_missing_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered columns needed by the saved model if missing."""
    df = df.copy()

    required_base_cols = ["quote_id", "product", "unit_price", "subtotal_price", "gross_margin_rate", "qty"]
    missing_base = [col for col in required_base_cols if col not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns: {missing_base}")

    # Raw competitor columns.
    for col in ["competitor_a", "competitor_b", "competitor_c"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Core numeric columns.
    df = to_numeric_if_exists(
        df,
        ["unit_price", "qty", "subtotal_price", "gross_margin_rate", "kw", "energy_grant_amount"],
    )

    if "energy_grant_amount" not in df.columns:
        df["energy_grant_amount"] = 0.0
    df["energy_grant_amount"] = df["energy_grant_amount"].fillna(0.0)

    # Competitor availability flags.
    for raw_col, flag_col in [
        ("competitor_a", "is_compe_a"),
        ("competitor_b", "is_compe_b"),
        ("competitor_c", "is_compe_c"),
    ]:
        if flag_col not in df.columns:
            df[flag_col] = df[raw_col].notna().astype(int)
        else:
            df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce")

    if "competitor_count_available" not in df.columns:
        df["competitor_count_available"] = df[["competitor_a", "competitor_b", "competitor_c"]].notna().sum(axis=1)
    else:
        df["competitor_count_available"] = pd.to_numeric(df["competitor_count_available"], errors="coerce")

    # Competitor summary prices.
    if "avg_competitor_price" not in df.columns:
        df["avg_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].mean(axis=1)
    if "min_competitor_price" not in df.columns:
        df["min_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].min(axis=1)
    if "max_competitor_price" not in df.columns:
        df["max_competitor_price"] = df[["competitor_a", "competitor_b", "competitor_c"]].max(axis=1)

    # Highest price helper, for duplicate cleaning compatibility.
    if "price_order" not in df.columns:
        df["price_order"] = (
            df.groupby(["quote_id", "product"])["unit_price"]
            .rank(method="dense", ascending=False)
            .astype("Int64")
        )
    if "is_highest_price" not in df.columns:
        df["is_highest_price"] = (df["price_order"] == 1).astype(int)

    # Initial cost/profit based on current price and current GMR.
    # Important: this current estimated_cost will be fixed during scenario simulation.
    if "estimated_cost" not in df.columns:
        df["estimated_cost"] = df["subtotal_price"] * (1 - df["gross_margin_rate"])
    else:
        df["estimated_cost"] = pd.to_numeric(df["estimated_cost"], errors="coerce")
        df["estimated_cost"] = df["estimated_cost"].where(
            df["estimated_cost"].notna(),
            df["subtotal_price"] * (1 - df["gross_margin_rate"]),
        )

    if "estimated_gross_profit" not in df.columns:
        df["estimated_gross_profit"] = df["subtotal_price"] - df["estimated_cost"]

    return recalculate_features_after_price_change(df)


def clean_for_modeling(
    df: pd.DataFrame,
    features: List[str],
    categorical_features: List[str],
    numeric_features: List[str],
) -> pd.DataFrame:
    """Replicate basic cleaning used for model training."""
    df_model = df.copy()

    before = len(df_model)
    price_variation = df_model.groupby(["quote_id", "product"])["unit_price"].transform("nunique")
    df_model = df_model[
        (price_variation == 1)
        | ((price_variation > 1) & (df_model["is_highest_price"] == 1))
    ].copy()
    print(f"Duplicate price cleaning: {before} -> {len(df_model)} rows")

    # Remove invalid qty rows, if any.
    if "qty" in df_model.columns:
        before_qty = len(df_model)
        df_model = df_model[df_model["qty"].notna() & (df_model["qty"] != 0)].copy()
        if before_qty != len(df_model):
            print(f"Removed zero/missing qty rows: {before_qty} -> {len(df_model)} rows")

    # Target.
    df_model = df_model[df_model["convert_to_order"].isin([0, 1])].copy()
    df_model["success"] = (df_model["convert_to_order"] == 0).astype(int)

    # Product grouping. Must match model training logic.
    product_counts = df_model["product"].value_counts()
    rare_products = product_counts[product_counts < MIN_PRODUCT_COUNT].index
    df_model["product_model"] = df_model["product"].replace(rare_products, "Other")

    # Ensure model features exist.
    for col in features:
        if col not in df_model.columns:
            df_model[col] = np.nan
            print(f"Warning: missing model feature '{col}' was created as NaN.")

    for col in numeric_features:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    for col in categorical_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype("string").fillna("Missing")

    return df_model.reset_index(drop=True)


# ============================================================
# Model prediction and GMR scenario simulation
# ============================================================

def predict_success_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Predict P(success=1), robust to class order."""
    proba = model.predict_proba(X)
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    classes = getattr(classifier, "classes_", None)

    if classes is not None and 1 in list(classes):
        positive_idx = list(classes).index(1)
    else:
        positive_idx = 1

    return proba[:, positive_idx]


def build_current_table(df_model: pd.DataFrame, model, features: List[str]) -> pd.DataFrame:
    current = df_model.copy().reset_index(drop=True)
    current["row_id"] = np.arange(len(current))

    current["current_predicted_win_probability"] = predict_success_probability(model, current[features].copy())
    current["current_predicted_win_probability_pct"] = current["current_predicted_win_probability"] * 100
    current["current_expected_return"] = current["current_predicted_win_probability"] * current["estimated_gross_profit"]

    keep_cols = [
        "row_id", "quote_id", "product", "product_model", "kw", "qty",
        "unit_price", "subtotal_price", "gross_margin_rate", "energy_grant_amount",
        "estimated_cost", "estimated_gross_profit", "avg_competitor_price", "min_competitor_price",
        "price_gap_avg_competitor_pct", "price_gap_min_competitor_pct",
        "convert_to_order", "success",
        "current_predicted_win_probability", "current_predicted_win_probability_pct",
        "current_expected_return",
    ]
    keep_cols = [col for col in keep_cols if col in current.columns]
    return current[keep_cols].copy()


def apply_candidate_gmr(base_df: pd.DataFrame, candidate_gmr: float) -> pd.DataFrame:
    """Apply one candidate GMR to all rows.

    Realistic assumption:
    - estimated_cost is fixed
    - subtotal/unit price changes to achieve candidate GMR
    - all derived price/profit/competitor features are recalculated
    """
    df = base_df.copy()
    candidate_gmr = float(candidate_gmr)

    estimated_cost = pd.to_numeric(df["estimated_cost"], errors="coerce")
    current_subtotal = pd.to_numeric(df["subtotal_price"], errors="coerce")
    current_gmr = pd.to_numeric(df["gross_margin_rate"], errors="coerce")

    # Fallback if estimated cost missing.
    estimated_cost = estimated_cost.where(
        estimated_cost.notna(),
        current_subtotal * (1 - current_gmr),
    )

    new_subtotal = estimated_cost / (1 - candidate_gmr)
    qty = pd.to_numeric(df["qty"], errors="coerce")
    current_unit = pd.to_numeric(df["unit_price"], errors="coerce")

    # Preferred calculation: subtotal / qty. Fallback to scaling current unit price.
    new_unit_by_qty = np.where(qty.notna() & (qty != 0), new_subtotal / qty, np.nan)
    scale = np.where(current_subtotal.notna() & (current_subtotal != 0), new_subtotal / current_subtotal, np.nan)
    new_unit_by_scale = current_unit * scale
    new_unit_price = np.where(pd.notna(new_unit_by_qty), new_unit_by_qty, new_unit_by_scale)

    df["candidate_gmr"] = candidate_gmr
    df["gross_margin_rate"] = candidate_gmr
    df["estimated_cost"] = estimated_cost
    df["subtotal_price"] = new_subtotal
    df["unit_price"] = new_unit_price

    return recalculate_features_after_price_change(df)


def create_all_scenarios(
    df_model: pd.DataFrame,
    gmr_grid: Iterable[float],
    categorical_features: List[str],
    numeric_features: List[str],
) -> pd.DataFrame:
    base = df_model.copy().reset_index(drop=True)
    base["row_id"] = np.arange(len(base))
    base["current_gmr"] = base["gross_margin_rate"]
    base["current_unit_price"] = base["unit_price"]
    base["current_subtotal_price"] = base["subtotal_price"]
    base["current_estimated_gross_profit"] = base["estimated_gross_profit"]

    scenario_parts = [apply_candidate_gmr(base, gmr) for gmr in gmr_grid]
    scenarios = pd.concat(scenario_parts, ignore_index=True)

    for col in numeric_features:
        if col in scenarios.columns:
            scenarios[col] = pd.to_numeric(scenarios[col], errors="coerce")
    for col in categorical_features:
        if col in scenarios.columns:
            scenarios[col] = scenarios[col].astype("string").fillna("Missing")

    return scenarios


def apply_optional_guardrails(
    scenarios: pd.DataFrame,
    max_gmr_change: Optional[float] = None,
    max_price_increase_pct: Optional[float] = None,
    max_price_gap_min_pct: Optional[float] = None,
) -> pd.DataFrame:
    """Apply optional business constraints to candidate scenarios.

    Parameters are optional. If not supplied, no guardrail is applied.
    """
    df = scenarios.copy()
    before = len(df)

    if max_gmr_change is not None:
        max_gmr_change = normalize_decimal_or_pct(max_gmr_change)
        df = df[(df["candidate_gmr"] - df["current_gmr"]).abs() <= max_gmr_change].copy()
        print(f"Guardrail max GMR change <= {max_gmr_change * 100:.2f} pp: {before:,} -> {len(df):,}")
        before = len(df)

    if max_price_increase_pct is not None:
        max_price_increase_pct = float(max_price_increase_pct)
        df["candidate_price_increase_pct"] = np.where(
            df["current_unit_price"].notna() & (df["current_unit_price"] != 0),
            ((df["unit_price"] - df["current_unit_price"]) / df["current_unit_price"]) * 100,
            np.nan,
        )
        df = df[df["candidate_price_increase_pct"].isna() | (df["candidate_price_increase_pct"] <= max_price_increase_pct)].copy()
        print(f"Guardrail max price increase <= {max_price_increase_pct:.2f}%: {before:,} -> {len(df):,}")
        before = len(df)

    if max_price_gap_min_pct is not None:
        max_price_gap_min_pct = float(max_price_gap_min_pct)
        # If min competitor is missing, keep the row because the constraint cannot be evaluated.
        df = df[
            df["price_gap_min_competitor_pct"].isna()
            | (df["price_gap_min_competitor_pct"] <= max_price_gap_min_pct)
        ].copy()
        print(f"Guardrail max gap vs min competitor <= {max_price_gap_min_pct:.2f}%: {before:,} -> {len(df):,}")

    return df.reset_index(drop=True)


def add_candidate_predictions(scenarios: pd.DataFrame, model, features: List[str]) -> pd.DataFrame:
    out = scenarios.copy()
    out["predicted_win_probability"] = predict_success_probability(model, out[features].copy())
    out["predicted_win_probability_pct"] = out["predicted_win_probability"] * 100
    out["candidate_estimated_gross_profit"] = out["estimated_gross_profit"]
    out["expected_return"] = out["predicted_win_probability"] * out["candidate_estimated_gross_profit"]
    return out


# ============================================================
# Threshold analysis
# ============================================================

def select_best_candidate(scenarios: pd.DataFrame, threshold: float) -> pd.DataFrame:
    eligible = scenarios[scenarios["predicted_win_probability"] >= threshold].copy()
    if eligible.empty:
        return eligible

    # Select max expected return. Tie-breakers: higher win prob, then lower GMR.
    eligible = eligible.sort_values(
        ["row_id", "expected_return", "predicted_win_probability", "candidate_gmr"],
        ascending=[True, False, False, True],
    )
    best = eligible.groupby("row_id", as_index=False).head(1).reset_index(drop=True)
    best["min_win_probability_threshold"] = threshold
    best["min_win_probability_threshold_pct"] = threshold * 100
    return best


def make_detail_table(best: pd.DataFrame, current_table: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()

    rec_keep = [
        "row_id", "candidate_gmr", "unit_price", "subtotal_price", "estimated_cost",
        "candidate_estimated_gross_profit", "predicted_win_probability",
        "predicted_win_probability_pct", "expected_return",
        "effective_price_after_grant", "grant_ratio_to_subtotal",
        "price_gap_avg_competitor_pct", "price_gap_min_competitor_pct",
        "higher_than_avg_competitor", "is_lower_than_competitor",
        "min_win_probability_threshold", "min_win_probability_threshold_pct",
    ]
    rec_keep = [col for col in rec_keep if col in best.columns]

    rec = best[rec_keep].copy().rename(columns={
        "candidate_gmr": "recommended_gmr",
        "unit_price": "recommended_unit_price",
        "subtotal_price": "recommended_subtotal_price",
        "candidate_estimated_gross_profit": "recommended_estimated_gross_profit",
        "predicted_win_probability": "recommended_predicted_win_probability",
        "predicted_win_probability_pct": "recommended_predicted_win_probability_pct",
        "expected_return": "recommended_expected_return",
        "effective_price_after_grant": "recommended_effective_price_after_grant",
        "grant_ratio_to_subtotal": "recommended_grant_ratio_to_subtotal",
        "price_gap_avg_competitor_pct": "recommended_price_gap_avg_competitor_pct",
        "price_gap_min_competitor_pct": "recommended_price_gap_min_competitor_pct",
    })

    out = current_table.merge(rec, on="row_id", how="inner")

    out["delta_gmr"] = out["recommended_gmr"] - out["gross_margin_rate"]
    out["delta_unit_price"] = out["recommended_unit_price"] - out["unit_price"]
    out["delta_subtotal_price"] = out["recommended_subtotal_price"] - out["subtotal_price"]
    out["delta_predicted_win_probability"] = (
        out["recommended_predicted_win_probability"] - out["current_predicted_win_probability"]
    )
    out["delta_expected_return"] = out["recommended_expected_return"] - out["current_expected_return"]

    out["current_gmr_pct"] = out["gross_margin_rate"] * 100
    out["recommended_gmr_pct"] = out["recommended_gmr"] * 100
    out["delta_gmr_pct_point"] = out["delta_gmr"] * 100
    out["delta_predicted_win_probability_pct_point"] = out["delta_predicted_win_probability"] * 100
    out["delta_expected_return_pct"] = np.where(
        out["current_expected_return"] != 0,
        (out["delta_expected_return"] / out["current_expected_return"]) * 100,
        np.nan,
    )

    first_cols = [
        "quote_id", "product", "product_model", "kw", "qty",
        "current_gmr_pct", "recommended_gmr_pct", "delta_gmr_pct_point",
        "unit_price", "recommended_unit_price", "delta_unit_price",
        "subtotal_price", "recommended_subtotal_price", "delta_subtotal_price",
        "estimated_cost", "estimated_gross_profit", "recommended_estimated_gross_profit",
        "current_predicted_win_probability_pct", "recommended_predicted_win_probability_pct",
        "delta_predicted_win_probability_pct_point",
        "current_expected_return", "recommended_expected_return", "delta_expected_return", "delta_expected_return_pct",
        "avg_competitor_price", "min_competitor_price",
        "price_gap_avg_competitor_pct", "recommended_price_gap_avg_competitor_pct",
        "price_gap_min_competitor_pct", "recommended_price_gap_min_competitor_pct",
        "convert_to_order", "success", "min_win_probability_threshold_pct",
    ]
    first_cols = [col for col in first_cols if col in out.columns]
    remaining_cols = [col for col in out.columns if col not in first_cols]
    return out[first_cols + remaining_cols].copy()


def summarize_threshold(threshold: float, detail: pd.DataFrame, current_table: pd.DataFrame) -> Dict[str, float]:
    total_rows = len(current_table)
    total_quote_ids = current_table["quote_id"].nunique() if "quote_id" in current_table.columns else np.nan
    current_return_all = current_table["current_expected_return"].sum()

    if detail.empty:
        return {
            "min_win_probability_threshold": threshold,
            "min_win_probability_threshold_pct": threshold * 100,
            "total_rows": total_rows,
            "total_quote_ids": total_quote_ids,
            "eligible_rows": 0,
            "eligible_quote_ids": 0,
            "eligible_rows_pct": 0,
            "eligible_quote_ids_pct": 0,
            "current_expected_return_all": current_return_all,
            "current_expected_return_eligible": 0,
            "recommended_expected_return_eligible": 0,
            "delta_expected_return_eligible": 0,
            "potential_return_lift_pct_eligible": np.nan,
            "potential_return_lift_pct_portfolio_conservative": 0,
            "avg_current_gmr_pct_eligible": np.nan,
            "avg_recommended_gmr_pct": np.nan,
            "avg_current_win_probability_pct_eligible": np.nan,
            "avg_recommended_win_probability_pct": np.nan,
            "avg_delta_win_probability_pct_point": np.nan,
            "avg_delta_gmr_pct_point": np.nan,
            "avg_delta_unit_price": np.nan,
            "avg_delta_subtotal_price": np.nan,
        }

    eligible_rows = len(detail)
    eligible_quote_ids = detail["quote_id"].nunique() if "quote_id" in detail.columns else np.nan

    current_return_eligible = detail["current_expected_return"].sum()
    recommended_return_eligible = detail["recommended_expected_return"].sum()
    delta = recommended_return_eligible - current_return_eligible

    return {
        "min_win_probability_threshold": threshold,
        "min_win_probability_threshold_pct": threshold * 100,
        "total_rows": total_rows,
        "total_quote_ids": total_quote_ids,
        "eligible_rows": eligible_rows,
        "eligible_quote_ids": eligible_quote_ids,
        "eligible_rows_pct": safe_pct(eligible_rows, total_rows),
        "eligible_quote_ids_pct": safe_pct(eligible_quote_ids, total_quote_ids),
        "current_expected_return_all": current_return_all,
        "current_expected_return_eligible": current_return_eligible,
        "recommended_expected_return_eligible": recommended_return_eligible,
        "delta_expected_return_eligible": delta,
        "potential_return_lift_pct_eligible": safe_pct(delta, current_return_eligible),
        "potential_return_lift_pct_portfolio_conservative": safe_pct(delta, current_return_all),
        "avg_current_gmr_pct_eligible": detail["current_gmr_pct"].mean(),
        "avg_recommended_gmr_pct": detail["recommended_gmr_pct"].mean(),
        "avg_current_win_probability_pct_eligible": detail["current_predicted_win_probability_pct"].mean(),
        "avg_recommended_win_probability_pct": detail["recommended_predicted_win_probability_pct"].mean(),
        "avg_delta_win_probability_pct_point": detail["delta_predicted_win_probability_pct_point"].mean(),
        "avg_delta_gmr_pct_point": detail["delta_gmr_pct_point"].mean(),
        "avg_delta_unit_price": detail["delta_unit_price"].mean(),
        "avg_delta_subtotal_price": detail["delta_subtotal_price"].mean(),
    }


# ============================================================
# Outputs
# ============================================================

def write_key_findings(summary: pd.DataFrame, output_path: str) -> None:
    lines = [
        "Potential expected return lift from realistic model-based GMR recommendation",
        "=======================================================================",
        "",
        "Interpretation note:",
        "The figures below are model-based expected return simulations, not realized profit.",
        "Expected return = predicted win probability × estimated gross profit.",
        "The realistic simulation keeps estimated cost fixed and recalculates price, subtotal, competitor gap, grant ratio, and gross profit for each candidate GMR.",
        "",
    ]

    for _, row in summary.iterrows():
        th = row["min_win_probability_threshold_pct"]
        eligible_rows = int(row["eligible_rows"])
        total_rows = int(row["total_rows"])
        coverage = row["eligible_rows_pct"]
        lift = row["potential_return_lift_pct_eligible"]
        portfolio_lift = row["potential_return_lift_pct_portfolio_conservative"]
        avg_gmr = row["avg_recommended_gmr_pct"]
        avg_win = row["avg_recommended_win_probability_pct"]

        if pd.isna(lift):
            lines.append(
                f"With a minimum predicted win probability of {th:.0f}%, no eligible recommendation was found from the tested GMR grid."
            )
        else:
            lines.append(
                f"With a minimum predicted win probability of {th:.0f}%, the realistic GMR recommender gives a potential expected return lift of "
                f"{lift:.2f}% among eligible quote/product rows ({eligible_rows:,}/{total_rows:,} rows, {coverage:.2f}% coverage). "
                f"Assuming ineligible rows remain unchanged, the conservative portfolio-level lift is {portfolio_lift:.2f}%. "
                f"The average recommended GMR is {avg_gmr:.2f}% with an average predicted win probability of {avg_win:.2f}%."
            )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_charts(summary: pd.DataFrame, output_dir: str) -> None:
    plot_df = summary.copy()
    plot_df["threshold_label"] = plot_df["min_win_probability_threshold_pct"].round(0).astype(int).astype(str) + "%"

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["threshold_label"], plot_df["potential_return_lift_pct_eligible"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Potential Expected Return Lift by Minimum Win Probability")
    plt.xlabel("Minimum predicted win probability")
    plt.ylabel("Potential expected return lift among eligible rows (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "realistic_potential_return_lift_by_threshold.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(plot_df["threshold_label"], plot_df["eligible_rows_pct"])
    plt.title("Recommendation Coverage by Minimum Win Probability")
    plt.xlabel("Minimum predicted win probability")
    plt.ylabel("Eligible quote/product rows (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "realistic_recommendation_coverage_by_threshold.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(plot_df["threshold_label"], plot_df["avg_recommended_gmr_pct"], marker="o")
    plt.title("Average Recommended GMR by Minimum Win Probability")
    plt.xlabel("Minimum predicted win probability")
    plt.ylabel("Average recommended GMR (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "realistic_avg_recommended_gmr_by_threshold.png"), dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate realistic potential expected return lift using saved GMR recommender model bundle."
    )
    parser.add_argument("--data", required=True, help="Path to df_preprocessed.csv")
    parser.add_argument("--bundle", default="output/final_gmr_recommender_bundle.joblib", help="Path to saved model bundle joblib")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--gmr-min", type=float, default=0.0, help="Minimum GMR candidate, decimal or percent")
    parser.add_argument("--gmr-max", type=float, default=0.60, help="Maximum GMR candidate, decimal or percent")
    parser.add_argument("--gmr-interval", type=float, default=0.05, help="GMR interval, decimal or percent")
    parser.add_argument("--thresholds", default="0.60,0.75,0.90", help="Comma-separated minimum win probability thresholds")
    parser.add_argument("--max-gmr-change", type=float, default=None, help="Optional max absolute GMR change from current GMR, decimal or percentage points, e.g. 0.15 or 15")
    parser.add_argument("--max-price-increase-pct", type=float, default=None, help="Optional max candidate unit price increase percentage vs current unit price, e.g. 20")
    parser.add_argument("--max-price-gap-min-pct", type=float, default=None, help="Optional max candidate price gap percentage vs minimum competitor price, e.g. 25")
    parser.add_argument("--save-all-scenarios", action="store_true", help="Save all candidate GMR scenarios; file can be large")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading trained model bundle...")
    bundle = load_bundle(args.bundle)
    model = bundle["model"]
    features = list(bundle["features"])
    categorical_features = list(bundle.get("categorical_features", ["product_model"]))
    numeric_features = list(bundle.get("numeric_features", [c for c in features if c not in categorical_features]))

    print(f"Model name: {bundle.get('model_name', type(model).__name__)}")
    print(f"Model config: {bundle.get('model_config', 'unknown')}")
    print(f"Number of model features: {len(features)}")

    print("Loading data...")
    df = load_data(args.data)
    print(f"Loaded shape: {df.shape}")

    print("Adding/checking feature engineering columns...")
    df = add_missing_feature_engineering(df)

    print("Preparing modeling dataset...")
    df_model = clean_for_modeling(df, features, categorical_features, numeric_features)
    print(f"Modeling shape: {df_model.shape}")
    print("Target distribution:")
    print(df_model["success"].map({0: "Fail", 1: "Success"}).value_counts(normalize=True).rename("proportion"))

    gmr_grid = build_gmr_grid(args.gmr_min, args.gmr_max, args.gmr_interval)
    thresholds = parse_float_list(args.thresholds)

    print("GMR grid:", ", ".join(f"{g * 100:.2f}%" for g in gmr_grid))
    print("Thresholds:", ", ".join(f"{t * 100:.0f}%" for t in thresholds))

    print("Predicting current baseline expected return...")
    current_table = build_current_table(df_model, model, features)

    print("Creating realistic GMR candidate scenarios...")
    scenarios = create_all_scenarios(df_model, gmr_grid, categorical_features, numeric_features)

    if any(x is not None for x in [args.max_gmr_change, args.max_price_increase_pct, args.max_price_gap_min_pct]):
        print("Applying optional business guardrails...")
        scenarios = apply_optional_guardrails(
            scenarios,
            max_gmr_change=args.max_gmr_change,
            max_price_increase_pct=args.max_price_increase_pct,
            max_price_gap_min_pct=args.max_price_gap_min_pct,
        )

    print("Predicting win probability for every realistic GMR candidate...")
    scenarios = add_candidate_predictions(scenarios, model, features)

    if args.save_all_scenarios:
        scenarios_path = os.path.join(args.output_dir, "realistic_all_gmr_candidate_scenarios.csv")
        scenarios.to_csv(scenarios_path, index=False)
        print(f"Saved all candidate scenarios: {scenarios_path}")

    print("Calculating threshold-based potential return lift...")
    summary_rows = []
    for threshold in thresholds:
        print(f"\nThreshold: {threshold * 100:.0f}%")
        best = select_best_candidate(scenarios, threshold)
        detail = make_detail_table(best, current_table)
        summary_row = summarize_threshold(threshold, detail, current_table)
        summary_rows.append(summary_row)

        label = threshold_label(threshold)
        detail_path = os.path.join(args.output_dir, f"realistic_recommendations_threshold_{label}.csv")
        detail.to_csv(detail_path, index=False)

        print(f"Eligible rows: {summary_row['eligible_rows']:,}/{summary_row['total_rows']:,} ({summary_row['eligible_rows_pct']:.2f}%)")
        if pd.notna(summary_row["potential_return_lift_pct_eligible"]):
            print(f"Potential expected return lift among eligible rows: {summary_row['potential_return_lift_pct_eligible']:.2f}%")
            print(f"Conservative portfolio-level lift: {summary_row['potential_return_lift_pct_portfolio_conservative']:.2f}%")
            print(f"Average recommended GMR: {summary_row['avg_recommended_gmr_pct']:.2f}%")
            print(f"Average recommended win probability: {summary_row['avg_recommended_win_probability_pct']:.2f}%")
        else:
            print("No eligible candidates for this threshold.")
        print(f"Saved detail: {detail_path}")

    summary = pd.DataFrame(summary_rows).sort_values("min_win_probability_threshold").reset_index(drop=True)

    rounded_summary = summary.copy()
    for col in rounded_summary.select_dtypes(include=["float", "float64", "float32"]).columns:
        rounded_summary[col] = rounded_summary[col].round(4)

    summary_path = os.path.join(args.output_dir, "realistic_potential_return_lift_summary.csv")
    rounded_summary.to_csv(summary_path, index=False)

    key_finding_path = os.path.join(args.output_dir, "realistic_potential_return_key_findings.txt")
    write_key_findings(summary, key_finding_path)

    save_charts(summary, args.output_dir)

    # Save run config for reproducibility.
    run_config = {
        "data": args.data,
        "bundle": args.bundle,
        "gmr_grid": [float(x) for x in gmr_grid],
        "thresholds": [float(x) for x in thresholds],
        "max_gmr_change": args.max_gmr_change,
        "max_price_increase_pct": args.max_price_increase_pct,
        "max_price_gap_min_pct": args.max_price_gap_min_pct,
        "model_name": bundle.get("model_name", type(model).__name__),
        "model_config": bundle.get("model_config", "unknown"),
        "features": features,
    }
    with open(os.path.join(args.output_dir, "realistic_potential_return_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\nDone.")
    print(f"Summary saved: {summary_path}")
    print(f"Key finding text saved: {key_finding_path}")
    print("Charts saved:")
    print(f"- {os.path.join(args.output_dir, 'realistic_potential_return_lift_by_threshold.png')}")
    print(f"- {os.path.join(args.output_dir, 'realistic_recommendation_coverage_by_threshold.png')}")
    print(f"- {os.path.join(args.output_dir, 'realistic_avg_recommended_gmr_by_threshold.png')}")

    display_cols = [
        "min_win_probability_threshold_pct",
        "eligible_rows",
        "eligible_rows_pct",
        "potential_return_lift_pct_eligible",
        "potential_return_lift_pct_portfolio_conservative",
        "avg_recommended_gmr_pct",
        "avg_recommended_win_probability_pct",
        "avg_delta_gmr_pct_point",
        "avg_delta_unit_price",
    ]
    display_cols = [c for c in display_cols if c in rounded_summary.columns]
    print("\nQuick summary:")
    print(rounded_summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

# python src/gmr_threshold_return_lift.py --data dataset/df_preprocessed.csv --bundle output/final_gmr_recommender_bundle.joblib --gmr-min 0 --gmr-max 0.60 --gmr-interval 0.05 --thresholds 0.60,0.75,0.90