#!/usr/bin/env python3
"""Train and compare win-percentage regressors from the saved CSV only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


def features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    rating_cols = [f"rating_{i}" for i in range(1, 11)]
    position_cols = [f"position_{i}" for i in range(1, 11)]
    return frame[rating_cols + position_cols].copy(), rating_cols, position_cols


def preprocessor(rating_cols: list[str], position_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "ratings",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                rating_cols,
            ),
            (
                "positions",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                position_cols,
            ),
        ]
    )


def model_specs() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=10.0),
        "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=20_000, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.025, max_depth=2, min_samples_leaf=8,
            loss="huber", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=600, max_depth=5, min_samples_leaf=5, max_features=0.75,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=600, max_depth=6, min_samples_leaf=4, max_features=0.85,
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
    }


def metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def run_tabfm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    *,
    backend: str,
) -> tuple[object, np.ndarray]:
    from tabfm import TabFMRegressor

    # TabFM accepts mixed pandas columns directly; no one-hot/scaling is applied.
    if backend == "pytorch":
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_release
    else:
        from tabfm import tabfm_v1_0_0_jax as tabfm_release
    foundation_model = tabfm_release.load(model_type="regression")
    model = TabFMRegressor(model=foundation_model, n_estimators=8, max_num_rows=240)
    model.fit(X_train, y_train)
    return model, np.asarray(model.predict(X_test), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nba_2k_team_seasons.csv")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--tabfm", choices=("auto", "jax", "pytorch", "skip"), default="auto")
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    required_rows = 30 * 10
    if len(frame) != required_rows:
        raise ValueError(f"Expected {required_rows} rows in CSV, found {len(frame)}")
    X, rating_cols, position_cols = features(frame)
    target = frame["win_pct"].to_numpy(dtype=float)
    train_mask = frame["season_end_year"] <= 2024
    test_mask = frame["season_end_year"] >= 2025
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = target[train_mask], target[test_mask]
    groups = frame.loc[train_mask, "season_end_year"].to_numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    predictions_out = frame.loc[test_mask, ["team", "season", "win_pct"]].copy()
    folds = GroupKFold(n_splits=4)
    fitted: dict[str, object] = {}

    baseline = np.full_like(y_test, y_train.mean())
    results.append({"model": "train_mean_baseline", "cv_rmse": np.nan, **metrics(y_test, baseline)})
    predictions_out["pred_train_mean_baseline"] = baseline

    for name, estimator in model_specs().items():
        pipeline = Pipeline([("prep", preprocessor(rating_cols, position_cols)), ("model", estimator)])
        scores = cross_val_score(
            pipeline, X_train, y_train, groups=groups, cv=folds,
            scoring="neg_root_mean_squared_error", n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        prediction = np.asarray(pipeline.predict(X_test), dtype=float)
        results.append({"model": name, "cv_rmse": float(-scores.mean()), **metrics(y_test, prediction)})
        predictions_out[f"pred_{name}"] = prediction
        fitted[name] = pipeline

    tabfm_error = None
    if args.tabfm != "skip":
        backends = ("jax", "pytorch") if args.tabfm == "auto" else (args.tabfm,)
        for backend in backends:
            try:
                tabfm_model, prediction = run_tabfm(X_train, y_train, X_test, backend=backend)
                results.append({"model": f"tabfm_{backend}", "cv_rmse": np.nan, **metrics(y_test, prediction)})
                predictions_out[f"pred_tabfm_{backend}"] = prediction
                fitted[f"tabfm_{backend}"] = tabfm_model
                tabfm_error = None
                break
            except Exception as exc:
                tabfm_error = f"{type(exc).__name__}: {exc}"

    results_frame = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    # Classical-model selection uses grouped CV only, never the held-out test score.
    classical = pd.DataFrame([r for r in results if pd.notna(r["cv_rmse"])])
    best_name = str(classical.sort_values("cv_rmse").iloc[0]["model"])
    joblib.dump(fitted[best_name], output_dir / "best_classical_model.joblib")
    results_frame.to_csv(output_dir / "model_metrics.csv", index=False)
    predictions_out.to_csv(output_dir / "test_predictions.csv", index=False)
    summary = {
        "csv": str(Path(args.data)),
        "train_seasons": sorted(frame.loc[train_mask, "season"].unique().tolist()),
        "test_seasons": sorted(frame.loc[test_mask, "season"].unique().tolist()),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "best_classical_model_by_grouped_cv": best_name,
        "tabfm_error": tabfm_error,
        "note": "TabFM pretrained weights are non-commercial/non-production licensed.",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results_frame.to_string(index=False))
    if tabfm_error:
        print(f"TabFM unavailable: {tabfm_error}")


if __name__ == "__main__":
    main()
