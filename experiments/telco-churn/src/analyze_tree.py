"""Compare tree split criteria, size limits, and cross-validation fold counts."""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from common import DATA_FILE, ROOT, make_preprocessor
from train import grid_specs, load_data


FOLDS = (3, 5, 10)


def candidate_summary(results: dict, index: int) -> dict:
    return {
        "params": results["params"][index],
        "cv_auc": float(results["mean_test_score"][index]),
        "fold_auc_std": float(results["std_test_score"][index]),
        "train_auc": float(results["mean_train_score"][index]),
    }


def main() -> None:
    X, y, _ = load_data(DATA_FILE)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    tree, grid = grid_specs(quick=False)["decision_tree"]
    summary = {
        "design": {
            "train_rows": len(X_train),
            "test_rows_held_out": len(X_test),
            "random_state": 42,
            "scoring": "roc_auc",
            "grid": grid,
            "note": "Exploratory comparison; the held-out test set is not used to select parameters or folds.",
        },
        "fold_comparison": [],
    }
    for folds in FOLDS:
        search = GridSearchCV(
            Pipeline([("preprocess", make_preprocessor()), ("model", tree)]),
            grid,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        print(f"Evaluating {len(ParameterGrid(grid))} candidates x {folds} folds", flush=True)
        search.fit(X_train, y_train)
        results = search.cv_results_
        best = candidate_summary(results, search.best_index_)
        best["actual_depth"] = search.best_estimator_.named_steps["model"].get_depth()
        best["actual_leaves"] = search.best_estimator_.named_steps["model"].get_n_leaves()
        record = {"folds": folds, "fits": len(ParameterGrid(grid)) * folds, "best": best}
        if folds == 5:
            record["criterion_best"] = {
                criterion: candidate_summary(
                    results,
                    max(
                        (i for i, p in enumerate(results["params"])
                         if p["model__criterion"] == criterion),
                        key=lambda i: results["mean_test_score"][i],
                    ),
                )
                for criterion in grid["model__criterion"]
            }
            record["controlled_comparisons"] = {}
            for key, values in grid.items():
                record["controlled_comparisons"][key] = []
                for value in values:
                    params = {**search.best_params_, key: value}
                    index = results["params"].index(params)
                    record["controlled_comparisons"][key].append(
                        {"value": value, **candidate_summary(results, index)}
                    )
        summary["fold_comparison"].append(record)
        print(f"{folds} folds: best CV AUC={best['cv_auc']:.5f}, params={best['params']}", flush=True)
    output = ROOT / "results" / "tree_parameter_analysis.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2,
                   default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
