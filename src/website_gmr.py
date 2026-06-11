"""
Minimal website / Streamlit usage example for the saved GMR recommender bundle.

This example assumes your website already prepares the same feature columns used
by the model. For raw quotation inputs, apply the same feature engineering first.
"""

import joblib
import pandas as pd


def load_recommender(bundle_path="output/final_gmr_recommender_bundle.joblib"):
    bundle = joblib.load(bundle_path)
    return bundle


def predict_win_probability(prepared_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predict current win probability using prepared feature columns."""
    model = bundle["model"]
    features = bundle["features"]

    missing = [col for col in features if col not in prepared_df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    out = prepared_df.copy()
    out["predicted_win_probability"] = model.predict_proba(out[features])[:, 1]
    out["predicted_win_probability_pct"] = (out["predicted_win_probability"] * 100).round(2)
    return out


if __name__ == "__main__":
    bundle = load_recommender("output/final_gmr_recommender_bundle.joblib")

    # Replace this with the dataframe produced by your web form + feature engineering.
    input_df = pd.read_csv("prepared_quote_features.csv")

    result = predict_win_probability(input_df, bundle)
    print(result[["predicted_win_probability", "predicted_win_probability_pct"]].head())
