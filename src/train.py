"""Reproduce the Bank Marketing classification experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_DATA = ROOT / "data" / "bank-additional-full.csv"
DEFAULT_OUTPUT = ROOT / "results"
SEED = 42


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset missing: {path}\nRun `python src/download_data.py` first."
        )
    frame = pd.read_csv(path, sep=";")
    if "y" not in frame.columns or set(frame["y"].dropna().unique()) != {"no", "yes"}:
        raise ValueError("Expected a semicolon-separated UCI bank-additional-full.csv")
    X = frame.drop(columns="y")
    y = frame["y"].map({"no": 0, "yes": 1}).astype(int)
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def model_specs(quick: bool) -> dict[str, tuple[object, dict[str, list]]]:
    specs = {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=2000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l1", "l2"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=120, class_weight="balanced_subsample", n_jobs=1, random_state=SEED),
            {"model__min_samples_leaf": [2, 5], "model__max_features": ["sqrt", 0.5]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", max_iter=120, random_state=SEED),
            {"model__max_leaf_nodes": [15, 31], "model__learning_rate": [0.05, 0.1]},
        ),
    }
    if quick:
        return {name: (model, {key: [values[0]] for key, values in grid.items()})
                for name, (model, grid) in specs.items()}
    return specs


def run(data_path: Path, output_dir: Path, folds: int, quick: bool) -> None:
    X, y = load_data(data_path)
    if quick:
        X, _, y, _ = train_test_split(X, y, train_size=min(6000, len(y) - 1),
                                       stratify=y, random_state=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Rows: {len(X):,}; positive: {y.mean():.2%}; train: {len(X_train):,}; test: {len(X_test):,}", flush=True)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    roc_data = {}

    for name, (model, grid) in model_specs(quick).items():
        print(f"\n=== {name}: {np.prod([len(v) for v in grid.values()])} candidates x {folds} folds ===", flush=True)
        pipe = Pipeline([("preprocess", make_preprocessor(X_train)), ("model", model)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True, verbose=0)
        search.fit(X_train, y_train)
        predicted = search.predict(X_test)
        probability = search.predict_proba(X_test)[:, 1]
        report_text = classification_report(y_test, predicted, target_names=["no", "yes"], zero_division=0)
        auc = roc_auc_score(y_test, probability)
        print(f"Best parameters: {search.best_params_}", flush=True)
        print(f"CV ROC AUC: {search.best_score_:.4f}; test ROC AUC: {auc:.4f}", flush=True)
        print(report_text, flush=True)
        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": float(search.best_score_),
            "test_roc_auc": float(auc),
            "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
            "classification_report": classification_report(
                y_test, predicted, target_names=["no", "yes"], zero_division=0, output_dict=True
            ),
        }
        roc_data[name] = roc_curve(y_test, probability)
        joblib.dump(search.best_estimator_, output_dir / f"{name}.joblib")

    (output_dir / "metrics.json").write_text(
        json.dumps({"dataset_rows": len(X), "test_rows": len(X_test),
                    "positive_rate": float(y.mean()), "cv_folds": folds,
                    "quick": quick, "models": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, result) in zip(axes, results.items()):
        ConfusionMatrixDisplay(np.array(result["confusion_matrix"]), display_labels=["no", "yes"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_title(name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, _) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC={results[name]['test_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test-set ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180)
    plt.close(fig)
    print(f"\nSaved outputs to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Smoke test with 6,000 rows and one grid point per model")
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    run(args.data, args.output, args.cv_folds, args.quick)


if __name__ == "__main__":
    main()
