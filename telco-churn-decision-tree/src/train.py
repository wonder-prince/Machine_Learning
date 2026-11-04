"""Train and evaluate a decision tree for IBM Telco Customer Churn."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from common import DATA_FILE, ROOT, clean_features, make_preprocessor

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    frame = pd.read_csv(path, dtype={"TotalCharges": "string"})
    if "Churn" not in frame or set(frame["Churn"].dropna().unique()) != {"No", "Yes"}:
        raise ValueError("Expected IBM Telco Customer Churn CSV with Churn = No/Yes")
    if frame["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found")
    X = clean_features(frame)
    y = frame["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    profile = {
        "rows": len(frame),
        "positive_rate": float(y.mean()),
        "missing_total_charges": int(X["TotalCharges"].isna().sum()),
        "features": list(X.columns),
    }
    return X, y, profile


def grid_specs(quick: bool) -> dict:
    tree_grid = {
        "model__criterion": ["gini", "entropy"],
        "model__max_depth": [3, 5, 7, 10],
        "model__max_leaf_nodes": [8, 16, 32, 64],
        "model__min_samples_leaf": [10, 30],
    }
    lr_grid = {"model__C": [0.01, 0.1, 1, 10]}
    if quick:
        tree_grid = {key: values[:1] for key, values in tree_grid.items()}
        lr_grid = {key: values[:1] for key, values in lr_grid.items()}
    return {
        "decision_tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42), tree_grid
        ),
        "logistic_regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
            lr_grid,
        ),
    }


def run(data_path: Path, output_dir: Path, folds: int, quick: bool) -> None:
    X, y, profile = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    metrics = {
        "dataset": profile,
        "split": {"train_rows": len(X_train), "test_rows": len(X_test),
                  "train_positive": int(y_train.sum()), "test_positive": int(y_test.sum()),
                  "random_state": 42, "cv_folds": folds},
        "quick": quick,
        "models": {},
    }
    rocs = {}
    for name, (estimator, grid) in grid_specs(quick).items():
        pipeline = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True
        )
        print(f"Training {name}: {len(ParameterGrid(grid))} candidates x {folds} folds", flush=True)
        search.fit(X_train, y_train)
        predicted = search.predict(X_test)
        probability = search.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probability)
        report = classification_report(
            y_test, predicted, labels=[0, 1], target_names=["No", "Yes"],
            zero_division=0,
        )
        print(f"\n{name}: best={search.best_params_}")
        print(f"CV AUC={search.best_score_:.4f}, test AUC={auc:.4f}")
        print(report)
        metrics["models"][name] = {
            "best_params": search.best_params_,
            "cv_auc": float(search.best_score_),
            "test_auc": float(auc),
            "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_test, predicted, labels=[0, 1], target_names=["No", "Yes"],
                zero_division=0, output_dict=True,
            ),
        }
        rocs[name] = roc_curve(y_test, probability)
        joblib.dump(search.best_estimator_, output_dir / f"{name}.joblib")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay(
        np.array(metrics["models"]["decision_tree"]["confusion_matrix"]),
        display_labels=["No", "Yes"],
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Decision tree: test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (fpr, tpr, _) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC={metrics['models'][name]['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=180)
    plt.close(fig)
    print(f"Saved results to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use one grid point per model")
    args = parser.parse_args()
    if args.cv_folds < 2:
        parser.error("--cv-folds must be at least 2")
    run(args.data, args.output, args.cv_folds, args.quick)


if __name__ == "__main__":
    main()
