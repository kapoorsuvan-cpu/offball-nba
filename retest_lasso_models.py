#!/usr/bin/env python3
"""Retest NBA win models with Lasso selection and starter-weighted features."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
ALPHAS = np.geomspace(0.05, 2.0, 60)
RIDGE_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
POSITION_CODES = ("PG", "SG", "SF", "PF", "C")


def canonical_position(value: str) -> str:
    value = str(value).upper().strip()
    aliases = {
        "G": "PG",
        "F": "SF",
        "G/F": "SG",
        "F/G": "SF",
        "F/C": "PF",
        "C/F": "C",
        "UNK": "SF",
    }
    if value in POSITION_CODES:
        return value
    return aliases.get(value, "SF")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    ratings = frame[[f"rating_{rank}" for rank in range(1, 11)]].astype(float)
    output = ratings.copy()

    top_five = ratings.iloc[:, :5].to_numpy()
    starter_core = ratings.iloc[:, 2:5].to_numpy()
    bench = ratings.iloc[:, 5:10].to_numpy()
    output["star_mean"] = ratings.iloc[:, :2].mean(axis=1)
    output["top_three_mean"] = ratings.iloc[:, :3].mean(axis=1)
    output["starter_mean"] = ratings.iloc[:, :5].mean(axis=1)
    output["starter_core_mean"] = ratings.iloc[:, 2:5].mean(axis=1)
    output["bench_mean"] = ratings.iloc[:, 5:10].mean(axis=1)
    output["top_eight_mean"] = ratings.iloc[:, :8].mean(axis=1)
    output["roster_mean"] = ratings.mean(axis=1)
    output["starter_weighted"] = top_five @ np.array([5, 4, 3, 2, 1]) / 15.0
    output["starter_core_weighted"] = starter_core @ np.array([3, 2, 1]) / 6.0
    output["bench_weighted"] = bench @ np.array([5, 4, 3, 2, 1]) / 15.0
    output["starter_floor"] = ratings["rating_5"]
    output["bench_floor"] = ratings["rating_10"]
    output["talent_spread"] = ratings["rating_1"] - ratings["rating_10"]
    output["star_to_bench_gap"] = output["star_mean"] - output["bench_mean"]
    output["elite_count"] = (ratings >= 85).sum(axis=1)
    output["strong_starter_count"] = (ratings.iloc[:, :5] >= 80).sum(axis=1)
    output["playable_depth_count"] = (ratings >= 75).sum(axis=1)

    # The 2K rating scale drifts across editions. These features compare each
    # roster with its 29 same-season peers, which are known at forecast time.
    season_groups = (
        frame["season_end_year"]
        if "season_end_year" in frame.columns
        else pd.Series(0, index=frame.index)
    )
    relative_columns = [
        *[f"rating_{rank}" for rank in range(1, 11)],
        "star_mean",
        "top_three_mean",
        "starter_mean",
        "starter_core_mean",
        "bench_mean",
        "top_eight_mean",
        "roster_mean",
        "starter_weighted",
        "starter_core_weighted",
        "bench_weighted",
        "starter_floor",
        "bench_floor",
        "talent_spread",
        "star_to_bench_gap",
        "elite_count",
        "strong_starter_count",
        "playable_depth_count",
    ]
    for column in relative_columns:
        group_mean = output[column].groupby(season_groups).transform("mean")
        group_std = output[column].groupby(season_groups).transform("std").replace(0, 1).fillna(1)
        output[f"relative_{column}"] = (output[column] - group_mean) / group_std

    explicit_prior_columns = all(
        f"prior_wins_{lag}" in frame.columns for lag in range(1, 4)
    )
    if explicit_prior_columns:
        for lag in range(1, 4):
            output[f"prior_wins_{lag}"] = pd.to_numeric(
                frame[f"prior_wins_{lag}"], errors="coerce"
            )
    elif {"team", "win_pct"}.issubset(frame.columns):
        normalized_wins = pd.to_numeric(frame["win_pct"], errors="coerce") * 82.0
        for lag in range(1, 4):
            output[f"prior_wins_{lag}"] = normalized_wins.groupby(frame["team"]).shift(lag)
    if "prior_wins_1" in output.columns:
        output["prior_three_year_mean"] = output[
            [f"prior_wins_{lag}" for lag in range(1, 4)]
        ].mean(axis=1)
        output["prior_year_trend"] = output["prior_wins_1"] - output["prior_wins_2"]

    position_values: dict[str, list[str]] = {}
    for rank in range(1, 11):
        values = frame[f"position_{rank}"].map(canonical_position)
        output[f"position_{rank}"] = values
        position_values[f"position_{rank}"] = values.tolist()
    for position in POSITION_CODES:
        output[f"starter_{position.lower()}_count"] = sum(
            (output[f"position_{rank}"] == position).astype(int) for rank in range(1, 6)
        )
        output[f"bench_{position.lower()}_count"] = sum(
            (output[f"position_{rank}"] == position).astype(int) for rank in range(6, 11)
        )
    return output


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [
        f"position_{rank}"
        for rank in range(1, 11)
        if f"position_{rank}" in frame.columns
    ]
    numeric = [column for column in frame.columns if column not in categorical]
    return numeric, categorical


def feature_view(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "full":
        return frame.copy()
    rank_positions = {f"position_{rank}" for rank in range(1, 11)}
    if mode == "no_rank_positions":
        return frame[[column for column in frame.columns if column not in rank_positions]].copy()
    if mode == "weighted_only":
        raw_ratings = {f"rating_{rank}" for rank in range(1, 11)}
        return frame[
            [
                column
                for column in frame.columns
                if column not in rank_positions and column not in raw_ratings
            ]
        ].copy()
    if mode == "relative_weighted":
        keep = [
            "relative_star_mean",
            "relative_top_three_mean",
            "relative_starter_mean",
            "relative_starter_core_mean",
            "relative_bench_mean",
            "relative_starter_weighted",
            "relative_starter_core_weighted",
            "relative_bench_weighted",
            "relative_starter_floor",
            "relative_bench_floor",
            "relative_talent_spread",
            "relative_star_to_bench_gap",
            "relative_elite_count",
            "relative_strong_starter_count",
            "relative_playable_depth_count",
            *[f"starter_{position.lower()}_count" for position in POSITION_CODES],
            *[f"bench_{position.lower()}_count" for position in POSITION_CODES],
        ]
        return frame[keep].copy()
    if mode == "relative_ratings":
        keep = [
            *[f"relative_rating_{rank}" for rank in range(1, 11)],
            "relative_starter_weighted",
            "relative_bench_weighted",
            "relative_starter_floor",
            "relative_bench_floor",
        ]
        return frame[keep].copy()
    if mode == "relative_ratings_history":
        keep = [
            *[f"relative_rating_{rank}" for rank in range(1, 11)],
            "relative_star_mean",
            "relative_top_three_mean",
            "relative_starter_mean",
            "relative_starter_core_mean",
            "relative_bench_mean",
            "relative_starter_weighted",
            "relative_bench_weighted",
            "relative_starter_floor",
            "relative_bench_floor",
            "prior_wins_1",
            "prior_wins_2",
            "prior_wins_3",
            "prior_three_year_mean",
            "prior_year_trend",
        ]
        return frame[keep].copy()
    if mode == "relative_ratings_history_compact":
        keep = [
            *[f"relative_rating_{rank}" for rank in range(1, 11)],
            "relative_star_mean",
            "relative_top_three_mean",
            "relative_starter_mean",
            "relative_top_eight_mean",
            "relative_roster_mean",
            "prior_wins_1",
            "prior_wins_2",
            "prior_wins_3",
            "prior_three_year_mean",
        ]
        return frame[keep].copy()
    raise ValueError(f"Unknown feature mode: {mode}")


def feature_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = feature_columns(frame)
    transformers = [
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
        ]
    if categorical:
        transformers.append(
            (
                "positions",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def make_pipeline(frame: pd.DataFrame, estimator: object) -> Pipeline:
    return Pipeline([("prep", feature_preprocessor(frame)), ("model", estimator)])


def grouped_rmse(
    factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> float:
    unique_groups = np.unique(groups)
    folds = GroupKFold(n_splits=min(5, len(unique_groups)))
    fold_scores = []
    for train_index, validation_index in folds.split(X, y, groups):
        model = factory()
        model.fit(X.iloc[train_index], y[train_index])
        prediction = model.predict(X.iloc[validation_index])
        fold_scores.append(math.sqrt(mean_squared_error(y[validation_index], prediction)))
    return float(np.mean(fold_scores))


def tune_parameter(
    values: list[float] | tuple[float, ...],
    factory: Callable[[float], Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, float]:
    scores = [(float(value), grouped_rmse(lambda value=value: factory(float(value)), X, y, groups)) for value in values]
    return min(scores, key=lambda item: item[1])


def metric_row(name: str, cv_rmse: float, actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | str]:
    errors = prediction - actual
    return {
        "model": name,
        "cv_rmse_wins": cv_rmse,
        "mae_wins": float(mean_absolute_error(actual, prediction)),
        "rmse_wins": float(math.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "prediction_std_wins": float(np.std(prediction)),
        "prediction_range_wins": float(np.ptp(prediction)),
        "actual_std_wins": float(np.std(actual)),
        "actual_range_wins": float(np.ptp(actual)),
        "mean_error_wins": float(np.mean(errors)),
    }


def current_feature_rows(
    app_data_path: Path,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    app_data = json.loads(app_data_path.read_text())
    aliases = {"LA Clippers": "Los Angeles Clippers"}
    rows = []
    teams = []
    for team in app_data["teams"]:
        players = team["players"][:10]
        row: dict[str, object] = {
            "team": aliases.get(team["name"], team["name"]),
            "season_end_year": 2027,
        }
        for rank, player in enumerate(players, start=1):
            row[f"rating_{rank}"] = player["rating"]
            row[f"position_{rank}"] = player["position"]
        prior_rows = history.loc[history["team"].eq(row["team"])].sort_values(
            "season_end_year", ascending=False
        )
        for lag in range(1, 4):
            row[f"prior_wins_{lag}"] = (
                float(prior_rows.iloc[lag - 1]["win_pct"]) * 82.0
                if len(prior_rows) >= lag
                else np.nan
            )
        rows.append(row)
        teams.append(team["name"])
    return engineer_features(pd.DataFrame(rows)), teams


def selected_lasso_features(model: Pipeline) -> list[dict[str, float | str]]:
    names = model.named_steps["prep"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
    selected = [
        {"feature": str(name), "coefficient_wins": float(coefficient)}
        for name, coefficient in zip(names, coefficients, strict=True)
        if abs(coefficient) > 1e-8
    ]
    return sorted(selected, key=lambda item: abs(float(item["coefficient_wins"])), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--app-data", default="dashboard-app/app/data/current-rosters.json")
    parser.add_argument("--output-dir", default="artifacts/lasso_retest")
    args = parser.parse_args()

    history = pd.read_csv(args.data)
    X = engineer_features(history)
    y = history["win_pct"].to_numpy(dtype=float) * 82.0
    train_mask = history["season_end_year"].le(2024).to_numpy()
    test_mask = history["season_end_year"].ge(2025).to_numpy()
    X_train, X_test = X.loc[train_mask].reset_index(drop=True), X.loc[test_mask].reset_index(drop=True)
    y_train, y_test = y[train_mask], y[test_mask]
    groups = history.loc[train_mask, "season_end_year"].to_numpy()

    lasso_modes: dict[str, dict[str, object]] = {}
    for mode in (
        "full",
        "no_rank_positions",
        "weighted_only",
        "relative_weighted",
        "relative_ratings",
        "relative_ratings_history",
        "relative_ratings_history_compact",
    ):
        X_mode = feature_view(X_train, mode)
        alpha, cv_rmse = tune_parameter(
            list(ALPHAS),
            lambda value, X_mode=X_mode: make_pipeline(
                X_mode,
                Lasso(alpha=value, max_iter=100_000, random_state=RANDOM_STATE),
            ),
            X_mode,
            y_train,
            groups,
        )
        lasso_modes[mode] = {"alpha": alpha, "cv_rmse": cv_rmse, "X": X_mode}
    lasso_alpha = float(lasso_modes["full"]["alpha"])
    lasso_cv = float(lasso_modes["full"]["cv_rmse"])
    ridge_alpha, ridge_cv = tune_parameter(
        RIDGE_ALPHAS,
        lambda alpha: make_pipeline(X_train, Ridge(alpha=alpha)),
        X_train,
        y_train,
        groups,
    )
    elastic_alpha, elastic_cv = tune_parameter(
        list(ALPHAS),
        lambda alpha: make_pipeline(
            X_train,
            ElasticNet(
                alpha=alpha,
                l1_ratio=0.75,
                max_iter=100_000,
                random_state=RANDOM_STATE,
            ),
        ),
        X_train,
        y_train,
        groups,
    )

    candidates: list[tuple[str, str, float, Pipeline]] = [
        (
            "lasso_engineered",
            "full",
            lasso_cv,
            make_pipeline(
                X_train,
                Lasso(alpha=lasso_alpha, max_iter=100_000, random_state=RANDOM_STATE),
            ),
        ),
        (
            "lasso_no_rank_positions",
            "no_rank_positions",
            float(lasso_modes["no_rank_positions"]["cv_rmse"]),
            make_pipeline(
                lasso_modes["no_rank_positions"]["X"],
                Lasso(
                    alpha=float(lasso_modes["no_rank_positions"]["alpha"]),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
        (
            "lasso_weighted_only",
            "weighted_only",
            float(lasso_modes["weighted_only"]["cv_rmse"]),
            make_pipeline(
                lasso_modes["weighted_only"]["X"],
                Lasso(
                    alpha=float(lasso_modes["weighted_only"]["alpha"]),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
        (
            "lasso_relative_weighted",
            "relative_weighted",
            float(lasso_modes["relative_weighted"]["cv_rmse"]),
            make_pipeline(
                lasso_modes["relative_weighted"]["X"],
                Lasso(
                    alpha=float(lasso_modes["relative_weighted"]["alpha"]),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
        (
            "lasso_relative_ratings",
            "relative_ratings",
            float(lasso_modes["relative_ratings"]["cv_rmse"]),
            make_pipeline(
                lasso_modes["relative_ratings"]["X"],
                Lasso(
                    alpha=float(lasso_modes["relative_ratings"]["alpha"]),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
        (
            "lasso_relative_ratings_history",
            "relative_ratings_history",
            float(lasso_modes["relative_ratings_history"]["cv_rmse"]),
            make_pipeline(
                lasso_modes["relative_ratings_history"]["X"],
                Lasso(
                    alpha=float(lasso_modes["relative_ratings_history"]["alpha"]),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
        (
            "lasso_relative_ratings_history_compact",
            "relative_ratings_history_compact",
            float(lasso_modes["relative_ratings_history_compact"]["cv_rmse"]),
            make_pipeline(
                lasso_modes["relative_ratings_history_compact"]["X"],
                Lasso(
                    alpha=float(lasso_modes["relative_ratings_history_compact"]["alpha"]),
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
        ("ridge_engineered", "full", ridge_cv, make_pipeline(X_train, Ridge(alpha=ridge_alpha))),
        (
            "elastic_net_engineered",
            "full",
            elastic_cv,
            make_pipeline(
                X_train,
                ElasticNet(
                    alpha=elastic_alpha,
                    l1_ratio=0.75,
                    max_iter=100_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ),
    ]

    extra_trees_factory = lambda: make_pipeline(
        X_train,
        ExtraTreesRegressor(
            n_estimators=800,
            max_depth=6,
            min_samples_leaf=4,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )
    candidates.append(("extra_trees_engineered", "full", grouped_rmse(extra_trees_factory, X_train, y_train, groups), extra_trees_factory()))

    rows = []
    fitted: dict[str, Pipeline] = {}
    feature_modes: dict[str, str] = {}
    predictions = history.loc[test_mask, ["team", "season", "season_end_year", "wins", "losses"]].reset_index(drop=True)
    for name, mode, cv_rmse, model in candidates:
        model.fit(feature_view(X_train, mode), y_train)
        prediction = np.clip(model.predict(feature_view(X_test, mode)), 5.0, 77.0)
        metric = metric_row(name, cv_rmse, y_test, prediction)
        metric["feature_mode"] = mode
        rows.append(metric)
        predictions[f"pred_{name}"] = prediction
        fitted[name] = model
        feature_modes[name] = mode

    metrics = pd.DataFrame(rows).sort_values("cv_rmse_wins").reset_index(drop=True)
    lasso_rows = metrics.loc[metrics["model"].str.startswith("lasso_")]
    selected_name = str(lasso_rows.sort_values("cv_rmse_wins").iloc[0]["model"])
    selected_mode = feature_modes[selected_name]

    selected_train_model = fitted[selected_name]
    selected_features = selected_lasso_features(selected_train_model)

    # Refit the chosen specification on all completed seasons for 2026-27.
    if selected_name.startswith("lasso_"):
        selected_alpha = float(lasso_modes[selected_mode]["alpha"])
        final_X = feature_view(X, selected_mode)
        final_model = make_pipeline(
            final_X,
            Lasso(alpha=selected_alpha, max_iter=100_000, random_state=RANDOM_STATE),
        )
    final_model.fit(final_X, y)

    current_X, current_teams = current_feature_rows(Path(args.app_data), history)
    current_predictions = pd.DataFrame(
        {
            "team": current_teams,
            "predicted_wins": np.clip(
                final_model.predict(feature_view(current_X, selected_mode)), 5.0, 77.0
            ),
        }
    ).sort_values("predicted_wins", ascending=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    predictions.to_csv(output_dir / "heldout_predictions.csv", index=False)
    current_predictions.to_csv(output_dir / "current_predictions.csv", index=False)
    joblib.dump(final_model, output_dir / "selected_model.joblib")
    summary = {
        "selectedModel": selected_name,
        "selectionRule": "Lowest grouped-CV RMSE among Lasso feature views.",
        "featureMode": selected_mode,
        "lassoAlpha": float(lasso_modes[selected_mode]["alpha"]),
        "lassoModes": {
            mode: {
                "alpha": float(values["alpha"]),
                "cvRmseWins": float(values["cv_rmse"]),
            }
            for mode, values in lasso_modes.items()
        },
        "ridgeAlpha": ridge_alpha,
        "elasticNetAlpha": elastic_alpha,
        "selectedFeatureCount": len(selected_features),
        "selectedFeatures": selected_features,
        "currentPredictionRangeWins": float(current_predictions["predicted_wins"].max() - current_predictions["predicted_wins"].min()),
        "currentPredictionStdWins": float(current_predictions["predicted_wins"].std(ddof=0)),
        "currentPredictionMin": current_predictions.iloc[-1].to_dict(),
        "currentPredictionMax": current_predictions.iloc[0].to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(metrics.to_string(index=False))
    print("\nSELECTED")
    print(json.dumps(summary, indent=2))
    print("\nCURRENT")
    print(current_predictions.to_string(index=False))


if __name__ == "__main__":
    main()
