"""Shared data cleaning and preprocessing for the churn experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "Telco-Customer-Churn.csv"
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
ORDINAL = ["Contract"]
CATEGORICAL = [column for column in FEATURES if column not in NUMERIC + ORDINAL]


def clean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the input schema and convert blank numeric values to missing."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input columns: {', '.join(missing)}")
    X = frame.loc[:, FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in CATEGORICAL + ORDINAL:
        X[column] = X[column].astype("string").fillna("missing").replace("", "missing").astype(str)
    return X


def make_preprocessor() -> ColumnTransformer:
    """Fit imputers and encoders inside each training fold."""
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("contract", Pipeline([
                ("encode", OrdinalEncoder(
                    categories=[["Month-to-month", "One year", "Two year"]],
                    handle_unknown="use_encoded_value", unknown_value=-1,
                )),
            ]), ORDINAL),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
