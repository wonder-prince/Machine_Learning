"""Predict churn for new customers from a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import ROOT, clean_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV with the 19 feature columns")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "predictions.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "results" / "decision_tree.joblib")
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"TotalCharges": "string"})
    if frame.empty:
        parser.error("Input CSV has no rows")
    model = joblib.load(args.model)
    probability = model.predict_proba(clean_features(frame))[:, 1]
    result = pd.DataFrame({
        "customerID": frame["customerID"] if "customerID" in frame else range(len(frame)),
        "churn_probability": probability,
        "predicted_churn": ["Yes" if p >= 0.5 else "No" for p in probability],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
