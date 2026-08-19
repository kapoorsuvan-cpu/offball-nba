#!/usr/bin/env python3
"""Run a reproducible generalization audit and freeze the current forecast.

The audit uses season-blocked resampling and strict walk-forward predictions.
No random row split is used because adjacent NBA seasons for the same franchise
are not independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from retest_lasso_models import engineer_features
from train_rank_calibrated_lasso import (
    FEATURES,
    calibrate_by_season,
    calibrate_one_season,
    make_lasso,
    rank_curve,
)


RANDOM_STATE = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(actual)),
        "maeWins": float(mean_absolute_error(actual, prediction)),
        "rmseWins": float(math.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "biasWins": float(np.mean(prediction - actual)),
        "predictionStdWins": float(np.std(prediction)),
        "actualStdWins": float(np.std(actual)),
        "predictionRangeWins": float(np.ptp(prediction)),
        "actualRangeWins": float(np.ptp(actual)),
    }


def add_baselines(history: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    ordered = history.sort_values(["team", "season_end_year"]).copy()
    ordered["normalized_wins"] = ordered["win_pct"] * 82.0
    ordered["prior_season_wins"] = ordered.groupby("team")["normalized_wins"].shift(1)
    ordered["prior_three_mean_wins"] = ordered.groupby("team")["normalized_wins"].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    baselines = ordered[
        ["team", "season_end_year", "prior_season_wins", "prior_three_mean_wins"]
    ]
    output = forecast.merge(
        baselines,
        on=["team", "season_end_year"],
        how="left",
        validate="one_to_one",
    )
    output["actual_normalized_wins"] = output["win_pct"] * 82.0
    output["league_mean_wins"] = 41.0
    return output


def baseline_tables(
    evaluated: pd.DataFrame,
    bets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = bets[
        ["team", "season_end_year", "win_total_line", "scheduled_games_for_line"]
    ].copy()
    market["market_wins_82"] = (
        market["win_total_line"] * 82.0 / market["scheduled_games_for_line"]
    )
    evaluated = evaluated.merge(
        market,
        on=["team", "season_end_year"],
        how="left",
        validate="one_to_one",
    )
    predictors = {
        "rank_calibrated_lasso": "predicted_wins",
        "prior_season": "prior_season_wins",
        "prior_three_seasons": "prior_three_mean_wins",
        "league_mean_41": "league_mean_wins",
        "preseason_market": "market_wins_82",
    }
    rows = []
    yearly_rows = []
    for name, column in predictors.items():
        valid = evaluated[column].notna()
        row = {"model": name, "scope": "all_10_seasons"}
        row.update(
            metrics(
                evaluated.loc[valid, "actual_normalized_wins"].to_numpy(float),
                evaluated.loc[valid, column].to_numpy(float),
            )
        )
        rows.append(row)
        recent = valid & evaluated["season_end_year"].ge(2025)
        recent_row = {"model": name, "scope": "2024-25_and_2025-26"}
        recent_row.update(
            metrics(
                evaluated.loc[recent, "actual_normalized_wins"].to_numpy(float),
                evaluated.loc[recent, column].to_numpy(float),
            )
        )
        rows.append(recent_row)
        for year, group in evaluated.loc[valid].groupby("season_end_year"):
            values = metrics(
                group["actual_normalized_wins"].to_numpy(float),
                group[column].to_numpy(float),
            )
            yearly_rows.append({"model": name, "seasonEndYear": int(year), **values})
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


def roster_continuity(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, team_rows in history.sort_values("season_end_year").groupby("team"):
        previous: set[str] | None = None
        for _, row in team_rows.iterrows():
            current = {str(row[f"player_{rank}"]) for rank in range(1, 11)}
            rows.append(
                {
                    "team": team,
                    "season_end_year": int(row["season_end_year"]),
                    "returningTopTen": len(current & previous) if previous is not None else np.nan,
                }
            )
            previous = current
    return pd.DataFrame(rows)


def residual_audit(
    history: pd.DataFrame,
    evaluated: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    residuals = evaluated.copy()
    residuals["errorWins"] = residuals["predicted_wins"] - residuals["actual_normalized_wins"]
    residuals["absoluteErrorWins"] = residuals["errorWins"].abs()
    residuals = residuals.merge(
        roster_continuity(history),
        on=["team", "season_end_year"],
        how="left",
        validate="one_to_one",
    )
    residuals["continuityTier"] = pd.cut(
        residuals["returningTopTen"],
        bins=[-0.1, 4, 7, 10],
        labels=["0-4 returning", "5-7 returning", "8-10 returning"],
    )
    residuals["actualWinTier"] = pd.cut(
        residuals["actual_normalized_wins"],
        bins=[-np.inf, 29.999, 49.999, np.inf],
        labels=["under 30", "30-49", "50+"],
    )
    segments = []
    for dimension in ("continuityTier", "actualWinTier", "season_end_year"):
        for value, group in residuals.groupby(dimension, observed=True):
            segments.append(
                {
                    "dimension": dimension,
                    "segment": str(value),
                    "rows": int(len(group)),
                    "maeWins": float(group["absoluteErrorWins"].mean()),
                    "rmseWins": float(math.sqrt(np.mean(group["errorWins"] ** 2))),
                    "biasWins": float(group["errorWins"].mean()),
                }
            )

    engineered = engineer_features(history)
    feature_frame = history[["team", "season_end_year"]].copy()
    for feature in FEATURES:
        feature_frame[feature] = engineered[feature]
    correlations = residuals.merge(
        feature_frame,
        on=["team", "season_end_year"],
        how="left",
        validate="one_to_one",
    )
    feature_correlations = {
        feature: {
            "errorCorrelation": float(correlations[feature].corr(correlations["errorWins"])),
            "absoluteErrorCorrelation": float(
                correlations[feature].corr(correlations["absoluteErrorWins"])
            ),
        }
        for feature in FEATURES
    }
    yearly = residuals.groupby("season_end_year")["absoluteErrorWins"].mean()
    trend = float(np.polyfit(yearly.index.to_numpy(float), yearly.to_numpy(float), 1)[0])
    summary = {
        "maeTrendWinsPerSeason": trend,
        "featureResidualCorrelations": feature_correlations,
        "worstAbsoluteErrorCorrelation": max(
            (
                {"feature": feature, **values}
                for feature, values in feature_correlations.items()
            ),
            key=lambda row: abs(row["absoluteErrorCorrelation"]),
        ),
    }
    return pd.DataFrame(segments), summary


def fit_gap_audit(
    history: pd.DataFrame,
    model_summary_path: Path,
) -> dict[str, Any]:
    """Compare an optimistic in-sample fit with chronological validation.

    The in-sample score intentionally reuses the same seasons to fit and score;
    it is not a performance claim. Its sole purpose is to make the classical
    train-versus-validation overfit gap explicit.
    """
    saved = json.loads(model_summary_path.read_text())
    alpha = float(saved["alpha"])
    blend = float(saved["rankBlend"])
    holdout_start = min(int(year) for year in saved["holdoutYears"])
    train_mask = history["season_end_year"].to_numpy(int) < holdout_start
    train_history = history.loc[train_mask].copy()
    engineered = engineer_features(history)
    target = history["win_pct"].to_numpy(float) * 82.0
    years = history["season_end_year"].to_numpy(int)
    model = make_lasso(alpha)
    model.fit(engineered.loc[train_mask, FEATURES], target[train_mask])
    base = np.asarray(model.predict(engineered.loc[train_mask, FEATURES]), dtype=float)
    fitted = calibrate_by_season(
        base,
        years[train_mask],
        rank_curve(target[train_mask], years[train_mask]),
        blend,
    )
    train_metrics = metrics(target[train_mask], fitted)
    validation = saved["expandingValidation"]
    holdout = saved["holdout"]
    return {
        "trainingSeasons": sorted(train_history["season"].unique().tolist()),
        "trainingInSample": train_metrics,
        "expandingValidation": validation,
        "finalTwoSeasonHoldout": holdout,
        "validationMinusTrainingRmseWins": float(
            validation["rmseWins"] - train_metrics["rmseWins"]
        ),
        "holdoutMinusValidationRmseWins": float(
            holdout["rmseWins"] - validation["rmseWins"]
        ),
        "interpretation": (
            "A small validation-minus-training gap argues against classical fit overfitting. "
            "A large holdout-minus-validation gap indicates temporal drift or a harder recent regime."
        ),
    }


def shuffled_target_test(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    repetitions: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    engineered = engineer_features(history)
    target = history["win_pct"].to_numpy(float) * 82.0
    years = history["season_end_year"].to_numpy(int)
    forecast_years = sorted(forecast["season_end_year"].unique())
    actual = forecast["win_pct"].to_numpy(float) * 82.0
    observed_rmse = float(
        math.sqrt(mean_squared_error(actual, forecast["predicted_wins"].to_numpy(float)))
    )
    rows = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(RANDOM_STATE + repetition)
        predictions = []
        for test_year in forecast_years:
            train_mask = years < test_year
            test_mask = years == test_year
            shuffled = target.copy()
            for train_year in np.unique(years[train_mask]):
                indices = np.flatnonzero(years == train_year)
                shuffled[indices] = rng.permutation(shuffled[indices])
            audit_row = forecast.loc[forecast["season_end_year"].eq(test_year)].iloc[0]
            model = make_lasso(float(audit_row["lasso_alpha"]))
            model.fit(engineered.loc[train_mask, FEATURES], shuffled[train_mask])
            base = np.asarray(model.predict(engineered.loc[test_mask, FEATURES]), dtype=float)
            predictions.extend(
                calibrate_one_season(
                    base,
                    rank_curve(shuffled[train_mask], years[train_mask]),
                    float(audit_row["rank_blend"]),
                )
            )
        prediction = np.asarray(predictions)
        row = {"repetition": repetition}
        row.update(metrics(actual, prediction))
        rows.append(row)
    frame = pd.DataFrame(rows)
    p_value = float((1 + frame["rmseWins"].le(observed_rmse).sum()) / (repetitions + 1))
    summary = {
        "repetitions": repetitions,
        "observedWalkForwardRmseWins": observed_rmse,
        "shuffledMedianRmseWins": float(frame["rmseWins"].median()),
        "shuffledBestRmseWins": float(frame["rmseWins"].min()),
        "oneSidedPermutationPValue": p_value,
        "passesAtFivePercent": bool(p_value < 0.05),
    }
    return frame, summary


def season_bootstrap(
    evaluated: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    years = np.sort(evaluated["season_end_year"].unique())
    predictors = {
        "rank_calibrated_lasso": "predicted_wins",
        "prior_season": "prior_season_wins",
        "prior_three_seasons": "prior_three_mean_wins",
        "league_mean_41": "league_mean_wins",
        "preseason_market": "market_wins_82",
    }
    rng = np.random.default_rng(RANDOM_STATE)
    samples: dict[str, list[float]] = {}
    for _ in range(repetitions):
        sampled_years = rng.choice(years, size=len(years), replace=True)
        sample = pd.concat(
            [evaluated.loc[evaluated["season_end_year"].eq(year)] for year in sampled_years],
            ignore_index=True,
        )
        actual = sample["actual_normalized_wins"].to_numpy(float)
        model_mae = float(mean_absolute_error(actual, sample["predicted_wins"]))
        model_rmse = float(
            math.sqrt(mean_squared_error(actual, sample["predicted_wins"]))
        )
        for name, column in predictors.items():
            valid = sample[column].notna()
            sample_actual = sample.loc[valid, "actual_normalized_wins"].to_numpy(float)
            prediction = sample.loc[valid, column].to_numpy(float)
            for metric_name, value in (
                ("maeWins", mean_absolute_error(sample_actual, prediction)),
                ("rmseWins", math.sqrt(mean_squared_error(sample_actual, prediction))),
                ("r2", r2_score(sample_actual, prediction)),
            ):
                samples.setdefault(f"{name}.{metric_name}", []).append(float(value))
            if name != "rank_calibrated_lasso":
                comparator_mae = float(mean_absolute_error(sample_actual, prediction))
                samples.setdefault(f"lassoMinus{name}.maeWins", []).append(
                    model_mae - comparator_mae
                )
                comparator_rmse = float(
                    math.sqrt(mean_squared_error(sample_actual, prediction))
                )
                samples.setdefault(f"lassoMinus{name}.rmseWins", []).append(
                    model_rmse - comparator_rmse
                )
    rows = []
    for name, values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "measure": name,
                "estimateMedian": float(np.median(array)),
                "ci95Low": float(np.quantile(array, 0.025)),
                "ci95High": float(np.quantile(array, 0.975)),
            }
        )
    return pd.DataFrame(rows).sort_values("measure")


def freeze_current_predictions(
    current_path: Path,
    model_path: Path,
    history_path: Path,
    app_data_path: Path,
    freeze_dir: Path,
    freeze_label: str,
) -> dict[str, Any]:
    app_data = json.loads(app_data_path.read_text())
    season = str(app_data["metadata"]["season"])
    freeze_date = str(app_data["metadata"]["generatedLocalDate"])
    freeze_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "-".join(freeze_label.lower().split())
    frozen_path = freeze_dir / f"{season}-{safe_label}-as-of-{freeze_date}.csv"
    current = pd.read_csv(current_path).sort_values("team").reset_index(drop=True)
    csv_bytes = current.to_csv(index=False).encode()
    if frozen_path.exists() and frozen_path.read_bytes() != csv_bytes:
        raise RuntimeError(f"Refusing to overwrite changed prediction freeze: {frozen_path}")
    if not frozen_path.exists():
        frozen_path.write_bytes(csv_bytes)
    manifest = {
        "season": season,
        "modelVersion": freeze_label,
        "frozenAsOf": freeze_date,
        "teamCount": int(len(current)),
        "predictionMinWins": float(current["predicted_wins"].min()),
        "predictionMaxWins": float(current["predicted_wins"].max()),
        "predictionRangeWins": float(current["predicted_wins"].max() - current["predicted_wins"].min()),
        "predictionFile": str(frozen_path),
        "predictionSha256": hashlib.sha256(csv_bytes).hexdigest(),
        "modelFile": str(model_path),
        "modelSha256": sha256(model_path),
        "historyFile": str(history_path),
        "historySha256": sha256(history_path),
        "appDataSha256": sha256(app_data_path),
        "evaluationRule": "Do not alter this file. Compare predicted_wins with final 2026-27 normalized wins after the regular season.",
    }
    manifest_path = freeze_dir / f"{season}-{safe_label}-as-of-{freeze_date}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument(
        "--forecast",
        default="artifacts/rank_lasso_walkforward_backtest/walkforward_predictions.csv",
    )
    parser.add_argument(
        "--bets", default="artifacts/rank_lasso_walkforward_backtest/all_bets.csv"
    )
    parser.add_argument(
        "--current", default="artifacts/rank_calibrated_lasso/current_predictions.csv"
    )
    parser.add_argument(
        "--model", default="artifacts/rank_calibrated_lasso/selected_model.joblib"
    )
    parser.add_argument(
        "--model-summary", default="artifacts/rank_calibrated_lasso/summary.json"
    )
    parser.add_argument("--app-data", default="dashboard-app/app/data/current-rosters.json")
    parser.add_argument("--output-dir", default="artifacts/generalization_audit")
    parser.add_argument("--freeze-dir", default="artifacts/prediction_freezes")
    parser.add_argument("--freeze-label", default="spread-calibrated-v2")
    parser.add_argument("--shuffle-repetitions", type=int, default=200)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    args = parser.parse_args()

    history_path = Path(args.history)
    forecast_path = Path(args.forecast)
    bets_path = Path(args.bets)
    current_path = Path(args.current)
    model_path = Path(args.model)
    model_summary_path = Path(args.model_summary)
    app_data_path = Path(args.app_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(history_path)
    forecast = pd.read_csv(forecast_path)
    bets = pd.read_csv(bets_path)
    evaluated = add_baselines(history, forecast)
    baseline_metrics, yearly_metrics = baseline_tables(evaluated, bets)
    market = bets[
        ["team", "season_end_year", "win_total_line", "scheduled_games_for_line"]
    ].copy()
    market["market_wins_82"] = (
        market["win_total_line"] * 82.0 / market["scheduled_games_for_line"]
    )
    evaluated = evaluated.merge(
        market,
        on=["team", "season_end_year"],
        how="left",
        validate="one_to_one",
    )
    residual_segments, residual_summary = residual_audit(history, evaluated)
    fit_gap_summary = fit_gap_audit(history, model_summary_path)
    shuffle_metrics, shuffle_summary = shuffled_target_test(
        history, forecast, args.shuffle_repetitions
    )
    bootstrap = season_bootstrap(evaluated, args.bootstrap_repetitions)
    freeze_manifest = freeze_current_predictions(
        current_path,
        model_path,
        history_path,
        app_data_path,
        Path(args.freeze_dir),
        args.freeze_label,
    )

    baseline_metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)
    yearly_metrics.to_csv(output_dir / "yearly_metrics.csv", index=False)
    residual_segments.to_csv(output_dir / "residual_segments.csv", index=False)
    shuffle_metrics.to_csv(output_dir / "shuffled_target_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "season_bootstrap_intervals.csv", index=False)
    evaluated.to_csv(output_dir / "evaluated_predictions.csv", index=False)

    overall = baseline_metrics.loc[
        baseline_metrics["scope"].eq("all_10_seasons")
    ].set_index("model")
    recent = baseline_metrics.loc[
        baseline_metrics["scope"].eq("2024-25_and_2025-26")
    ].set_index("model")
    market_delta = bootstrap.loc[
        bootstrap["measure"].eq("lassoMinuspreseason_market.maeWins")
    ].iloc[0]
    summary = {
        "method": "Strict walk-forward by season; season-block bootstrap; no random row split.",
        "walkForwardSeasons": sorted(forecast["season"].unique().tolist()),
        "modelOverall": overall.loc["rank_calibrated_lasso"].to_dict(),
        "modelRecentHoldout": recent.loc["rank_calibrated_lasso"].to_dict(),
        "baselineComparison": {
            name: overall.loc[name].to_dict()
            for name in [
                "prior_season",
                "prior_three_seasons",
                "league_mean_41",
                "preseason_market",
            ]
        },
        "marketMaeDifferenceBootstrap95": {
            "estimateMedian": float(market_delta["estimateMedian"]),
            "ci95Low": float(market_delta["ci95Low"]),
            "ci95High": float(market_delta["ci95High"]),
            "interpretation": "Negative favors the model; an interval crossing zero is inconclusive.",
        },
        "fitGapAudit": fit_gap_summary,
        "shuffledTargetTest": shuffle_summary,
        "residualAudit": residual_summary,
        "predictionFreeze": freeze_manifest,
        "conclusionRules": {
            "classicalOverfit": "Validation RMSE materially above training RMSE.",
            "leakage": "Shuffled-target model performs unusually close to the real model.",
            "temporalDrift": "Recent walk-forward errors materially exceed earlier errors.",
            "compression": "Prediction standard deviation is materially below actual standard deviation.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
