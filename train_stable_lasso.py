#!/usr/bin/env python3
"""Train a small, time-validated Lasso roster model.

The model intentionally uses only three roster tiers plus prior-season form.
Hyperparameters are selected with expanding-window validation so no fold trains
on a season later than the season it predicts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from assemble_lasso_app_data import current_raw_rows
from retest_lasso_models import engineer_features


RANDOM_STATE = 42
VALIDATION_YEARS = tuple(range(2018, 2025))
HOLDOUT_START_YEAR = 2025
ALPHAS = np.geomspace(0.1, 3.0, 60)
RIDGE_ALPHAS = np.geomspace(1.0, 300.0, 40)
STABLE_FEATURES = [
    "relative_star_mean",
    "relative_starter_core_mean",
    "relative_bench_mean",
    "prior_wins_1",
]


def make_model(estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def expanding_predictions(
    frame: pd.DataFrame,
    target: np.ndarray,
    years: np.ndarray,
    estimator_factory: Any,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    actual: list[float] = []
    predicted: list[float] = []
    coefficients: list[np.ndarray] = []
    for validation_year in VALIDATION_YEARS:
        train_mask = years < validation_year
        validation_mask = years == validation_year
        model = make_model(estimator_factory())
        model.fit(frame.loc[train_mask, STABLE_FEATURES], target[train_mask])
        predicted.extend(model.predict(frame.loc[validation_mask, STABLE_FEATURES]))
        actual.extend(target[validation_mask])
        coefficients.append(np.asarray(model.named_steps["model"].coef_, dtype=float))
    return np.asarray(actual), np.asarray(predicted), coefficients


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "maeWins": float(mean_absolute_error(actual, predicted)),
        "rmseWins": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "predictionStdWins": float(np.std(predicted)),
        "predictionRangeWins": float(np.ptp(predicted)),
    }


def tune(
    frame: pd.DataFrame,
    target: np.ndarray,
    years: np.ndarray,
    model_name: str,
) -> tuple[float, dict[str, float], list[np.ndarray]]:
    values = RIDGE_ALPHAS if model_name == "ridge" else ALPHAS
    candidates = []
    for value in values:
        if model_name == "lasso":
            factory = lambda value=value: Lasso(
                alpha=float(value), max_iter=100_000, random_state=RANDOM_STATE
            )
        elif model_name == "elastic_net":
            factory = lambda value=value: ElasticNet(
                alpha=float(value),
                l1_ratio=0.75,
                max_iter=100_000,
                random_state=RANDOM_STATE,
            )
        else:
            factory = lambda value=value: Ridge(alpha=float(value))
        actual, predicted, coefficients = expanding_predictions(
            frame, target, years, factory
        )
        score = metrics(actual, predicted)
        candidates.append((float(value), score, coefficients))
    # MAE is the primary user-facing accuracy measure; RMSE breaks near ties.
    return min(candidates, key=lambda item: (item[1]["maeWins"], item[1]["rmseWins"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--app-data", default="dashboard-app/app/data/current-rosters.json")
    parser.add_argument("--output-dir", default="artifacts/stable_lasso")
    args = parser.parse_args()

    history = pd.read_csv(args.history)
    engineered = engineer_features(history)
    target = history["win_pct"].to_numpy(dtype=float) * 82.0
    years = history["season_end_year"].to_numpy(dtype=int)
    train_mask = years < HOLDOUT_START_YEAR
    holdout_mask = years >= HOLDOUT_START_YEAR

    tuned: dict[str, dict[str, Any]] = {}
    for model_name in ("lasso", "elastic_net", "ridge"):
        parameter, validation, fold_coefficients = tune(
            engineered, target, years, model_name
        )
        tuned[model_name] = {
            "parameter": parameter,
            "validation": validation,
            "foldCoefficients": fold_coefficients,
        }

    for model_name, values in tuned.items():
        parameter = float(values["parameter"])
        if model_name == "lasso":
            estimator = Lasso(
                alpha=parameter, max_iter=100_000, random_state=RANDOM_STATE
            )
        elif model_name == "elastic_net":
            estimator = ElasticNet(
                alpha=parameter,
                l1_ratio=0.75,
                max_iter=100_000,
                random_state=RANDOM_STATE,
            )
        else:
            estimator = Ridge(alpha=parameter)
        comparison_model = make_model(estimator)
        comparison_model.fit(
            engineered.loc[train_mask, STABLE_FEATURES], target[train_mask]
        )
        values["holdout"] = metrics(
            target[holdout_mask],
            comparison_model.predict(
                engineered.loc[holdout_mask, STABLE_FEATURES]
            ),
        )

    selected_alpha = float(tuned["lasso"]["parameter"])
    selected = make_model(
        Lasso(alpha=selected_alpha, max_iter=100_000, random_state=RANDOM_STATE)
    )
    selected.fit(engineered.loc[train_mask, STABLE_FEATURES], target[train_mask])
    train_prediction = selected.predict(engineered.loc[train_mask, STABLE_FEATURES])
    holdout_prediction = selected.predict(engineered.loc[holdout_mask, STABLE_FEATURES])
    train_metrics = metrics(target[train_mask], train_prediction)
    holdout_metrics = metrics(target[holdout_mask], holdout_prediction)

    final_model = make_model(
        Lasso(alpha=selected_alpha, max_iter=100_000, random_state=RANDOM_STATE)
    )
    final_model.fit(engineered[STABLE_FEATURES], target)
    final_coefficients = np.asarray(final_model.named_steps["model"].coef_, dtype=float)

    data = json.loads(Path(args.app_data).read_text())
    current_raw = current_raw_rows(data, history)
    current_engineered = engineer_features(current_raw)
    current_prediction = np.clip(
        final_model.predict(current_engineered[STABLE_FEATURES]), 8.0, 74.0
    )
    current = pd.DataFrame(
        {
            "team": [team["name"] for team in data["teams"]],
            "predicted_wins": current_prediction,
        }
    ).sort_values("predicted_wins", ascending=False)

    fold_coefficients = np.asarray(tuned["lasso"]["foldCoefficients"])
    coefficient_rows = [
        {
            "feature": feature,
            "finalCoefficientWins": float(coefficient),
            "expandingMeanCoefficientWins": float(fold_coefficients[:, index].mean()),
            "expandingCoefficientStdWins": float(fold_coefficients[:, index].std()),
            "signConsistency": float(
                np.mean(np.sign(fold_coefficients[:, index]) == np.sign(coefficient))
            )
            if coefficient != 0
            else 1.0,
            "retained": bool(abs(coefficient) > 1e-8),
        }
        for index, (feature, coefficient) in enumerate(
            zip(STABLE_FEATURES, final_coefficients, strict=True)
        )
    ]

    comparison_rows = []
    for model_name, values in tuned.items():
        comparison_rows.append(
            {
                "model": model_name,
                "parameter": float(values["parameter"]),
                "validationMaeWins": values["validation"]["maeWins"],
                "validationRmseWins": values["validation"]["rmseWins"],
                "validationR2": values["validation"]["r2"],
                "holdoutMaeWins": values["holdout"]["maeWins"],
                "holdoutRmseWins": values["holdout"]["rmseWins"],
                "holdoutR2": values["holdout"]["r2"],
            }
        )

    holdout = history.loc[
        holdout_mask, ["team", "season", "season_end_year", "wins", "losses"]
    ].copy()
    holdout["predicted_wins"] = holdout_prediction
    holdout["error_wins"] = holdout_prediction - target[holdout_mask]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_dir / "selected_model.joblib")
    pd.DataFrame(comparison_rows).sort_values("validationMaeWins").to_csv(
        output_dir / "model_metrics.csv", index=False
    )
    holdout.to_csv(output_dir / "heldout_predictions.csv", index=False)
    current.to_csv(output_dir / "current_predictions.csv", index=False)
    summary = {
        "selectedModel": "stable_lasso_three_tier",
        "selectionRule": "Lowest expanding-window MAE; RMSE breaks ties.",
        "validationYears": list(VALIDATION_YEARS),
        "holdoutYears": [2025, 2026],
        "alpha": selected_alpha,
        "candidateFeatureCount": len(STABLE_FEATURES),
        "retainedFeatureCount": sum(row["retained"] for row in coefficient_rows),
        "features": coefficient_rows,
        "train": train_metrics,
        "expandingValidation": tuned["lasso"]["validation"],
        "holdout": holdout_metrics,
        "trainValidationRmseGapWins": float(
            tuned["lasso"]["validation"]["rmseWins"] - train_metrics["rmseWins"]
        ),
        "currentPredictionStdWins": float(np.std(current_prediction)),
        "currentPredictionRangeWins": float(np.ptp(current_prediction)),
        "currentPredictionMin": current.iloc[-1].to_dict(),
        "currentPredictionMax": current.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nMODEL COMPARISON")
    print(pd.DataFrame(comparison_rows).sort_values("validationMaeWins").to_string(index=False))
    print("\nCURRENT")
    print(current.to_string(index=False))


if __name__ == "__main__":
    main()
