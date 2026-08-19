#!/usr/bin/env python3
"""Walk-forward backtest for the rank-calibrated starter Lasso model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from backtest_win_totals import (
    BET_END_YEARS,
    COVID_VOID_END_YEAR,
    STANDARD_SEASON_GAMES,
    american_win_profit,
    binomial_upper_tail,
    grade_pick,
    parse_odds_json,
    scheduled_games_for_line,
    wilson_interval,
)
from retest_lasso_models import engineer_features
from train_rank_calibrated_lasso import (
    ALPHAS,
    BLENDS,
    FEATURES,
    MIN_VALIDATION_SPREAD_RATIO,
    calibrate_one_season,
    make_lasso,
    rank_curve,
)


def tune_temporal_alpha(
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray,
) -> tuple[float, float, float, float, float]:
    unique_years = np.sort(np.unique(years))
    validation_years = unique_years[1:]
    candidates: list[tuple[float, float, float, float, float]] = []
    for alpha in ALPHAS:
        actual: list[float] = []
        base_predictions: list[float] = []
        rank_predictions: list[float] = []
        for validation_year in validation_years:
            train_mask = years < validation_year
            validation_mask = years == validation_year
            model = make_lasso(float(alpha))
            model.fit(X.loc[train_mask, FEATURES], y[train_mask])
            fold_base = np.asarray(
                model.predict(X.loc[validation_mask, FEATURES]), dtype=float
            )
            curve = rank_curve(y[train_mask], years[train_mask])
            fold_ranks = np.argsort(np.argsort(-fold_base))
            base_predictions.extend(fold_base)
            rank_predictions.extend(curve[fold_ranks])
            actual.extend(y[validation_mask])
        actual_array = np.asarray(actual)
        base_array = np.asarray(base_predictions)
        rank_array = np.asarray(rank_predictions)
        for blend in BLENDS:
            predicted_array = (1.0 - blend) * base_array + blend * rank_array
            candidates.append(
                (
                    float(alpha),
                    float(blend),
                    float(np.mean(np.abs(predicted_array - actual_array))),
                    float(math.sqrt(mean_squared_error(actual_array, predicted_array))),
                    float(np.std(predicted_array) / np.std(actual_array)),
                )
            )
    eligible = [
        item for item in candidates if item[4] >= MIN_VALIDATION_SPREAD_RATIO
    ]
    if not eligible:
        raise RuntimeError("No walk-forward candidate passed the spread floor")
    return min(eligible, key=lambda item: (item[3], item[2]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--odds-json", default="data/historical_win_totals.browser.json")
    parser.add_argument("--output-dir", default="artifacts/rank_lasso_walkforward_backtest")
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--american-odds", type=int, default=-110)
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings)
    X_all = engineer_features(ratings)
    y_all = ratings["win_pct"].to_numpy(dtype=float) * 82.0
    predictions = []
    yearly_model_audit = []

    for end_year in BET_END_YEARS:
        train_mask = ratings["season_end_year"].lt(end_year).to_numpy()
        test_mask = ratings["season_end_year"].eq(end_year).to_numpy()
        X_train = X_all.loc[train_mask].reset_index(drop=True)
        y_train = y_all[train_mask]
        train_years = ratings.loc[train_mask, "season_end_year"].to_numpy()
        alpha, rank_blend, cv_mae, cv_rmse, cv_spread_ratio = tune_temporal_alpha(
            X_train,
            y_train,
            train_years,
        )
        model = make_lasso(alpha)
        model.fit(X_train[FEATURES], y_train)
        base_prediction = np.asarray(
            model.predict(X_all.loc[test_mask, FEATURES]), dtype=float
        )
        prediction = np.clip(
            calibrate_one_season(
                base_prediction,
                rank_curve(y_train, train_years),
                rank_blend,
            ),
            5.0,
            77.0,
        )
        season_rows = ratings.loc[
            test_mask,
            ["team", "season", "season_end_year", "wins", "losses", "win_pct"],
        ].copy()
        season_rows["predicted_wins"] = prediction
        season_rows["predicted_win_pct"] = prediction / 82.0
        season_rows["selected_model"] = "rank_calibrated_lasso"
        season_rows["lasso_alpha"] = alpha
        season_rows["rank_blend"] = rank_blend
        season_rows["training_seasons"] = int(ratings.loc[train_mask, "season_end_year"].nunique())
        predictions.append(season_rows)
        yearly_model_audit.append(
            {
                "season": str(season_rows.iloc[0]["season"]),
                "seasonEndYear": end_year,
                "lassoAlpha": alpha,
                "rankBlend": rank_blend,
                "temporalCvMaeWins": cv_mae,
                "groupedCvRmseWins": cv_rmse,
                "validationSpreadRatio": cv_spread_ratio,
                "predictionStdWins": float(np.std(prediction)),
                "predictionRangeWins": float(np.ptp(prediction)),
            }
        )

    forecast = pd.concat(predictions, ignore_index=True)
    odds = parse_odds_json(Path(args.odds_json))
    joined = forecast.merge(
        odds,
        on=["team", "season", "season_end_year"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 300:
        raise ValueError(f"Expected 300 joined forecast/odds rows, got {len(joined)}")

    # `predicted_wins` is the model's stable 82-game-equivalent target. Convert
    # it to the schedule covered by each preseason line only for betting and
    # market-error evaluation. In particular, 2020-21 lines covered 72 games.
    joined["predicted_wins_82"] = joined["predicted_wins"]
    joined["scheduled_games_for_line"] = joined["season_end_year"].map(
        scheduled_games_for_line
    )
    joined["predicted_wins_for_line"] = (
        joined["predicted_wins_82"]
        * joined["scheduled_games_for_line"]
        / STANDARD_SEASON_GAMES
    )
    joined["pick"] = np.where(
        joined["predicted_wins_for_line"] > joined["win_total_line"], "over", "under"
    )
    joined["model_edge_wins"] = (
        joined["predicted_wins_for_line"] - joined["win_total_line"]
    )
    joined["comparison_actual_wins"] = np.where(
        joined["season_end_year"].eq(COVID_VOID_END_YEAR),
        joined["win_pct"] * STANDARD_SEASON_GAMES,
        joined["wins"],
    )
    joined["model_abs_error_wins"] = (
        joined["predicted_wins_for_line"] - joined["comparison_actual_wins"]
    ).abs()
    joined["market_abs_error_wins"] = (
        joined["win_total_line"] - joined["comparison_actual_wins"]
    ).abs()
    joined["grade"] = [
        "void" if year == COVID_VOID_END_YEAR else grade_pick(pick, wins, line)
        for year, pick, wins, line in zip(
            joined["season_end_year"],
            joined["pick"],
            joined["wins"],
            joined["win_total_line"],
            strict=True,
        )
    ]
    win_profit = american_win_profit(args.stake, args.american_odds)
    joined["stake"] = args.stake
    joined["profit"] = joined["grade"].map(
        {"win": win_profit, "loss": -args.stake, "push": 0.0, "void": 0.0}
    )
    joined["cash_return"] = args.stake + joined["profit"]

    yearly_rows = []
    for (season, end_year), group in joined.groupby(["season", "season_end_year"], sort=True):
        settled = group["grade"].isin(["win", "loss"])
        model_errors = group["predicted_wins_for_line"] - group["comparison_actual_wins"]
        market_errors = group["win_total_line"] - group["comparison_actual_wins"]
        yearly_rows.append(
            {
                "season": season,
                "season_end_year": end_year,
                "bets": len(group),
                "wins": int(group["grade"].eq("win").sum()),
                "losses": int(group["grade"].eq("loss").sum()),
                "pushes": int(group["grade"].eq("push").sum()),
                "voids": int(group["grade"].eq("void").sum()),
                "win_rate_settled": float(group.loc[settled, "grade"].eq("win").mean()) if settled.any() else np.nan,
                "amount_staked": float(group["stake"].sum()),
                "settled_stake": float(settled.sum() * args.stake),
                "net_profit": float(group["profit"].sum()),
                "roi_on_settled_stake": float(group["profit"].sum() / (settled.sum() * args.stake)) if settled.any() else np.nan,
                "model_mae_wins": float(np.abs(model_errors).mean()),
                "market_mae_wins": float(np.abs(market_errors).mean()),
                "model_rmse_wins": float(math.sqrt(np.mean(model_errors**2))),
                "market_rmse_wins": float(math.sqrt(np.mean(market_errors**2))),
            }
        )
    yearly = pd.DataFrame(yearly_rows)
    yearly["cumulative_profit"] = yearly["net_profit"].cumsum()

    settled = joined["grade"].isin(["win", "loss"])
    wins = int(joined["grade"].eq("win").sum())
    losses = int(joined["grade"].eq("loss").sum())
    settled_bets = int(settled.sum())
    interval_low, interval_high = wilson_interval(wins, settled_bets)
    breakeven = abs(args.american_odds) / (abs(args.american_odds) + 100)
    overall = {
        "bets": len(joined),
        "settled_bets": settled_bets,
        "wins": wins,
        "losses": losses,
        "pushes": int(joined["grade"].eq("push").sum()),
        "voids": int(joined["grade"].eq("void").sum()),
        "win_rate": wins / settled_bets,
        "win_rate_wilson_95_low": interval_low,
        "win_rate_wilson_95_high": interval_high,
        "assumed_breakeven_win_rate": breakeven,
        "one_sided_binomial_p_vs_assumed_breakeven": binomial_upper_tail(wins, settled_bets, breakeven),
        "amount_staked": float(joined["stake"].sum()),
        "settled_stake": float(settled.sum() * args.stake),
        "net_profit": float(joined["profit"].sum()),
        "roi_on_settled_stake": float(joined["profit"].sum() / (settled.sum() * args.stake)),
        "overall_model_mae_wins": float(joined["model_abs_error_wins"].mean()),
        "overall_market_mae_wins": float(joined["market_abs_error_wins"].mean()),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.sort_values(["season_end_year", "team"]).to_csv(output_dir / "all_bets.csv", index=False)
    forecast.sort_values(["season_end_year", "team"]).to_csv(
        output_dir / "walkforward_predictions.csv", index=False
    )
    yearly.to_csv(output_dir / "yearly_summary.csv", index=False)
    (output_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2) + "\n")
    (output_dir / "model_audit.json").write_text(json.dumps(yearly_model_audit, indent=2) + "\n")
    assumptions = {
        "model": "rank_calibrated_lasso",
        "walkForward": True,
        "hyperparameterTuning": "Expanding-window season validation inside each training window",
        "features": FEATURES,
        "starterWeighting": "Lasso selects among star, starter, bench, and prior-form inputs inside each training window",
        "rankCalibration": (
            "Blend and alpha tuned with expanding-window seasons inside each training window; "
            f"candidates below {MIN_VALIDATION_SPREAD_RATIO:.0%} of observed spread are rejected"
        ),
        "stakePerTeam": args.stake,
        "assumedAmericanOdds": args.american_odds,
        "covid2019_20": "All bets voided because the scheduled season was shortened; stakes returned.",
        "shortened2020_21": (
            "The model remains trained on 82-game-equivalent wins, then predictions are "
            "scaled to 72 games before comparison with 2020-21 lines and final wins."
        ),
    }
    (output_dir / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")

    print(yearly.to_string(index=False))
    print("\nTOTAL")
    print(json.dumps(overall, indent=2))
    print("\nMODEL AUDIT")
    print(json.dumps(yearly_model_audit, indent=2))


if __name__ == "__main__":
    main()
