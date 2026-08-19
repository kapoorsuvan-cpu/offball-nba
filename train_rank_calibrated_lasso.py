#!/usr/bin/env python3
"""Train a starter-weighted Lasso with anti-compression calibration.

The Lasso chooses among ten compact roster/history inputs. A small calibration
blend maps predicted team rank toward the historical NBA win distribution.
Candidate models must also retain a minimum share of the observed league spread,
so average-error optimization cannot collapse every team toward 41 wins. Alpha
and blend are selected only from expanding-window seasons before the final
2024-25/2025-26 holdout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from assemble_lasso_app_data import current_raw_rows
from retest_lasso_models import engineer_features
from train_models import run_tabfm


RANDOM_STATE = 42
VALIDATION_YEARS = tuple(range(2018, 2025))
HOLDOUT_START_YEAR = 2025
FEATURES = [
    "relative_star_mean",
    "relative_top_three_mean",
    "relative_starter_mean",
    "relative_starter_core_mean",
    "relative_bench_mean",
    "relative_starter_floor",
    "relative_bench_floor",
    "relative_talent_spread",
    "prior_wins_1",
    "prior_three_year_mean",
]
ALPHAS = np.geomspace(0.05, 1.4, 60)
BLENDS = np.linspace(0.0, 0.60, 61)
MIN_VALIDATION_SPREAD_RATIO = 0.82


def make_lasso(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                Lasso(
                    alpha=float(alpha),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def rank_curve(target: np.ndarray, years: np.ndarray) -> np.ndarray:
    seasons = []
    for year in np.unique(years):
        values = np.sort(target[years == year])[::-1]
        if len(values) != 30:
            raise ValueError(f"Expected 30 teams for {year}, found {len(values)}")
        seasons.append(values)
    return np.mean(seasons, axis=0)


def calibrate_one_season(
    base_prediction: np.ndarray,
    curve: np.ndarray,
    blend: float,
) -> np.ndarray:
    ranks = np.argsort(np.argsort(-base_prediction))
    return (1.0 - blend) * base_prediction + blend * curve[ranks]


def calibrate_by_season(
    base_prediction: np.ndarray,
    years: np.ndarray,
    curve: np.ndarray,
    blend: float,
) -> np.ndarray:
    output = np.empty_like(base_prediction, dtype=float)
    for year in np.unique(years):
        mask = years == year
        output[mask] = calibrate_one_season(base_prediction[mask], curve, blend)
    return np.clip(output, 5.0, 77.0)


def metric_row(
    model: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    validation_rmse: float | None,
) -> dict[str, Any]:
    prediction_std = float(np.std(predicted))
    actual_std = float(np.std(actual))
    return {
        "model": model,
        "validationRmseWins": validation_rmse,
        "maeWins": float(mean_absolute_error(actual, predicted)),
        "rmseWins": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "predictionStdWins": prediction_std,
        "predictionRangeWins": float(np.ptp(predicted)),
        "actualStdWins": actual_std,
        "actualRangeWins": float(np.ptp(actual)),
        "spreadRatio": float(prediction_std / actual_std) if actual_std else 0.0,
    }


def tail_bias(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    low = actual < 30.0
    high = actual >= 50.0
    return {
        "under30BiasWins": float(np.mean(error[low])),
        "fiftyPlusBiasWins": float(np.mean(error[high])),
    }


def expanding_base_predictions(
    X: pd.DataFrame,
    target: np.ndarray,
    years: np.ndarray,
    factory: Callable[[], Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actual: list[float] = []
    base: list[float] = []
    ranked: list[float] = []
    for validation_year in VALIDATION_YEARS:
        train_mask = years < validation_year
        validation_mask = years == validation_year
        model = factory()
        model.fit(X.loc[train_mask, FEATURES], target[train_mask])
        fold_base = np.asarray(
            model.predict(X.loc[validation_mask, FEATURES]), dtype=float
        )
        curve = rank_curve(target[train_mask], years[train_mask])
        actual.extend(target[validation_mask])
        base.extend(fold_base)
        ranked.extend(curve[np.argsort(np.argsort(-fold_base))])
    return np.asarray(actual), np.asarray(base), np.asarray(ranked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--app-data", default="dashboard-app/app/data/current-rosters.json")
    parser.add_argument("--output-dir", default="artifacts/rank_calibrated_lasso")
    parser.add_argument("--tabfm", action="store_true")
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    engineered = engineer_features(history)
    target = history["win_pct"].to_numpy(dtype=float) * 82.0
    years = history["season_end_year"].to_numpy(dtype=int)
    train_mask = years < HOLDOUT_START_YEAR
    holdout_mask = years >= HOLDOUT_START_YEAR

    tuned: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        actual, base, ranked = expanding_base_predictions(
            engineered,
            target,
            years,
            lambda alpha=alpha: make_lasso(float(alpha)),
        )
        for blend in BLENDS:
            prediction = (1.0 - blend) * base + blend * ranked
            tuned.append(
                {
                    "alpha": float(alpha),
                    "blend": float(blend),
                    "mae": float(mean_absolute_error(actual, prediction)),
                    "rmse": float(math.sqrt(mean_squared_error(actual, prediction))),
                    "spreadRatio": float(np.std(prediction) / np.std(actual)),
                    "actual": actual,
                    "prediction": prediction,
                }
            )
    eligible = [
        row for row in tuned
        if row["spreadRatio"] >= MIN_VALIDATION_SPREAD_RATIO
    ]
    if not eligible:
        raise RuntimeError("No model candidate passed the anti-compression spread floor")
    selected = min(eligible, key=lambda row: (row["rmse"], row["mae"]))
    alpha = float(selected["alpha"])
    blend = float(selected["blend"])

    holdout_model = make_lasso(alpha)
    holdout_model.fit(engineered.loc[train_mask, FEATURES], target[train_mask])
    holdout_base = np.asarray(
        holdout_model.predict(engineered.loc[holdout_mask, FEATURES]), dtype=float
    )
    training_curve = rank_curve(target[train_mask], years[train_mask])
    holdout_prediction = calibrate_by_season(
        holdout_base,
        years[holdout_mask],
        training_curve,
        blend,
    )

    final_model = make_lasso(alpha)
    final_model.fit(engineered[FEATURES], target)
    final_curve = rank_curve(target, years)
    bundle = {
        "pipeline": final_model,
        "features": FEATURES,
        "rankCurve": final_curve,
        "rankBlend": blend,
        "name": "Starter Lasso + spread calibration",
        "minimumValidationSpreadRatio": MIN_VALIDATION_SPREAD_RATIO,
    }

    data = json.loads(Path(args.app_data).read_text())
    current_raw = current_raw_rows(data, history)
    current_engineered = engineer_features(current_raw)
    current_base = np.asarray(
        final_model.predict(current_engineered[FEATURES]), dtype=float
    )
    current_prediction = calibrate_one_season(
        current_base,
        final_curve,
        blend,
    )
    current = pd.DataFrame(
        {
            "team": [team["name"] for team in data["teams"]],
            "predicted_wins": current_prediction,
        }
    ).sort_values("predicted_wins", ascending=False)

    comparison_rows = [
        metric_row(
            "rank_calibrated_lasso",
            target[holdout_mask],
            holdout_prediction,
            validation_rmse=float(selected["rmse"]),
        )
    ]

    elastic_actual, elastic_base, _ = expanding_base_predictions(
        engineered,
        target,
        years,
        lambda: Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=0.4,
                        l1_ratio=0.7,
                        max_iter=100_000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    )
    elastic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=0.4,
                    l1_ratio=0.7,
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    elastic.fit(engineered.loc[train_mask, FEATURES], target[train_mask])
    elastic_prediction = elastic.predict(engineered.loc[holdout_mask, FEATURES])
    comparison_rows.append(
        metric_row(
            "elastic_net",
            target[holdout_mask],
            elastic_prediction,
            validation_rmse=float(
                math.sqrt(mean_squared_error(elastic_actual, elastic_base))
            ),
        )
    )

    extra_factory = lambda: Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=600,
                    max_depth=5,
                    min_samples_leaf=6,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    extra_actual, extra_base, _ = expanding_base_predictions(
        engineered, target, years, extra_factory
    )
    extra = extra_factory()
    extra.fit(engineered.loc[train_mask, FEATURES], target[train_mask])
    extra_prediction = extra.predict(engineered.loc[holdout_mask, FEATURES])
    comparison_rows.append(
        metric_row(
            "extra_trees",
            target[holdout_mask],
            extra_prediction,
            validation_rmse=float(math.sqrt(mean_squared_error(extra_actual, extra_base))),
        )
    )

    tabfm_error = None
    if args.tabfm:
        try:
            _, tabfm_prediction = run_tabfm(
                engineered.loc[train_mask, FEATURES].reset_index(drop=True),
                target[train_mask],
                engineered.loc[holdout_mask, FEATURES].reset_index(drop=True),
                backend="jax",
            )
            comparison_rows.append(
                metric_row(
                    "tabfm_jax",
                    target[holdout_mask],
                    tabfm_prediction,
                    validation_rmse=None,
                )
            )
        except Exception as exc:
            tabfm_error = f"{type(exc).__name__}: {exc}"

    final_coefficients = np.asarray(final_model.named_steps["model"].coef_, dtype=float)
    feature_rows = [
        {
            "feature": feature,
            "coefficientWins": float(coefficient),
            "retained": bool(abs(coefficient) > 1e-8),
        }
        for feature, coefficient in zip(FEATURES, final_coefficients, strict=True)
    ]
    validation_metrics = metric_row(
        "rank_calibrated_lasso",
        np.asarray(selected["actual"]),
        np.asarray(selected["prediction"]),
        validation_rmse=float(selected["rmse"]),
    )
    holdout_metrics = comparison_rows[0]
    summary = {
        "selectedModel": "rank_calibrated_lasso",
        "selectionRule": (
            "Lowest expanding-window RMSE among candidates retaining at least "
            f"{MIN_VALIDATION_SPREAD_RATIO:.0%} of observed validation spread; MAE breaks ties."
        ),
        "validationYears": list(VALIDATION_YEARS),
        "holdoutYears": [2025, 2026],
        "alpha": alpha,
        "rankBlend": blend,
        "candidateFeatureCount": len(FEATURES),
        "retainedFeatureCount": sum(row["retained"] for row in feature_rows),
        "features": feature_rows,
        "expandingValidation": validation_metrics,
        "holdout": holdout_metrics,
        "minimumValidationSpreadRatio": MIN_VALIDATION_SPREAD_RATIO,
        "validationTailBias": tail_bias(
            np.asarray(selected["actual"]), np.asarray(selected["prediction"])
        ),
        "holdoutTailBias": tail_bias(target[holdout_mask], holdout_prediction),
        "currentPredictionStdWins": float(np.std(current_prediction)),
        "currentPredictionRangeWins": float(np.ptp(current_prediction)),
        "currentPredictionMin": current.iloc[-1].to_dict(),
        "currentPredictionMax": current.iloc[0].to_dict(),
        "tabfmError": tabfm_error,
        "licenseNote": "TabFM pretrained weights are non-commercial/non-production licensed.",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "selected_model.joblib")
    pd.DataFrame(comparison_rows).to_csv(output_dir / "model_metrics.csv", index=False)
    current.to_csv(output_dir / "current_predictions.csv", index=False)
    holdout = history.loc[
        holdout_mask, ["team", "season", "season_end_year", "wins", "losses"]
    ].copy()
    holdout["predicted_wins"] = holdout_prediction
    holdout.to_csv(output_dir / "heldout_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("\nMODEL COMPARISON")
    print(pd.DataFrame(comparison_rows).to_string(index=False))
    print("\nCURRENT")
    print(current.to_string(index=False))


if __name__ == "__main__":
    main()
