#!/usr/bin/env python3
"""Walk-forward NBA win-total backtest using browser-pulled historical lines."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline

from train_models import features, model_specs, preprocessor


BET_END_YEARS = tuple(range(2017, 2027))
COVID_VOID_END_YEAR = 2020
STANDARD_SEASON_GAMES = 82
SCHEDULED_GAMES_BY_END_YEAR = {
    2021: 72,
}


def scheduled_games_for_line(season_end_year: int) -> int:
    """Return the number of games covered by that season's preseason line."""
    return SCHEDULED_GAMES_BY_END_YEAR.get(int(season_end_year), STANDARD_SEASON_GAMES)


def normalize_team(team: str) -> str:
    aliases = {"LA Clippers": "Los Angeles Clippers"}
    return aliases.get(team, team)


def american_win_profit(stake: float, odds: int) -> float:
    if odds > 0:
        return stake * odds / 100.0
    return stake * 100.0 / abs(odds)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return center - margin, center + margin


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    return sum(
        math.comb(trials, k) * probability**k * (1 - probability) ** (trials - k)
        for k in range(successes, trials + 1)
    )


def parse_odds_json(path: Path) -> pd.DataFrame:
    records = json.loads(path.read_text())
    rows = []
    for record in records:
        match = re.fullmatch(r"(\d+)-(\d+)\s+\(([^)]+)\)", record["result_text"])
        if not match:
            raise ValueError(f"Unexpected result text: {record['result_text']}")
        end_year = int(record["season_end_year"])
        rows.append(
            {
                "season": f"{end_year - 1}-{str(end_year)[-2:]}",
                "season_end_year": end_year,
                "team": normalize_team(record["team"]),
                "win_total_line": float(record["win_total_line"]),
                "actual_wins_odds_source": int(match.group(1)),
                "actual_losses_odds_source": int(match.group(2)),
                "source_result": match.group(3),
                "championship_odds": record["championship_odds"],
                "source_url": (
                    "https://www.basketball-reference.com/leagues/"
                    f"NBA_{end_year}_preseason_odds.html"
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["season_end_year", "team"]).reset_index(drop=True)
    if frame.shape[0] != 300 or frame.groupby("season_end_year").size().ne(30).any():
        raise ValueError("Expected exactly 30 odds rows for each of 10 seasons")
    return frame


def grade_pick(pick: str, actual_wins: float, line: float) -> str:
    if math.isclose(actual_wins, line):
        return "push"
    actual_side = "over" if actual_wins > line else "under"
    return "win" if pick == actual_side else "loss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--odds-json", default="data/historical_win_totals.browser.json")
    parser.add_argument("--odds-csv", default="data/nba_historical_win_totals.csv")
    parser.add_argument("--output-dir", default="artifacts/win_total_backtest")
    parser.add_argument("--stake", type=float, default=10.0)
    parser.add_argument("--american-odds", type=int, default=-110)
    parser.add_argument(
        "--model", choices=tuple(model_specs()) + ("walk_forward_cv",), default="extra_trees"
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings)
    odds = parse_odds_json(Path(args.odds_json))
    Path(args.odds_csv).parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(args.odds_csv, index=False)

    if ratings["season_end_year"].min() > 2014:
        raise ValueError("Backtest requires at least a three-season burn-in beginning in 2013-14")
    X_all, rating_cols, position_cols = features(ratings)
    predictions = []
    for end_year in BET_END_YEARS:
        train_mask = ratings["season_end_year"] < end_year
        test_mask = ratings["season_end_year"] == end_year
        if train_mask.sum() < 90 or test_mask.sum() != 30:
            raise ValueError(f"Invalid walk-forward split for {end_year}")
        selected_model = args.model
        if args.model == "walk_forward_cv":
            groups = ratings.loc[train_mask, "season_end_year"].to_numpy()
            cv = GroupKFold(n_splits=min(4, len(np.unique(groups))))
            cv_scores = {}
            for candidate_name, candidate in model_specs().items():
                candidate_pipeline = Pipeline(
                    [
                        ("prep", preprocessor(rating_cols, position_cols)),
                        ("model", candidate),
                    ]
                )
                scores = cross_val_score(
                    candidate_pipeline,
                    X_all.loc[train_mask],
                    ratings.loc[train_mask, "win_pct"],
                    groups=groups,
                    cv=cv,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=1,
                )
                cv_scores[candidate_name] = float(-scores.mean())
            selected_model = min(cv_scores, key=cv_scores.get)
        # A fresh estimator is constructed each year; only prior outcomes enter fit().
        estimator = model_specs()[selected_model]
        pipeline = Pipeline(
            [("prep", preprocessor(rating_cols, position_cols)), ("model", estimator)]
        )
        pipeline.fit(X_all.loc[train_mask], ratings.loc[train_mask, "win_pct"])
        predicted_pct = np.clip(pipeline.predict(X_all.loc[test_mask]), 0.0, 1.0)
        season_rows = ratings.loc[
            test_mask, ["team", "season", "season_end_year", "wins", "losses", "win_pct"]
        ].copy()
        season_rows["predicted_win_pct"] = predicted_pct
        season_rows["training_seasons"] = int(ratings.loc[train_mask, "season_end_year"].nunique())
        season_rows["selected_model"] = selected_model
        predictions.append(season_rows)

    forecast = pd.concat(predictions, ignore_index=True)
    joined = forecast.merge(
        odds,
        on=["team", "season", "season_end_year"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != 300:
        missing = forecast.merge(
            odds, on=["team", "season", "season_end_year"], how="left", indicator=True
        ).query("_merge != 'both'")
        raise ValueError(f"Odds join produced {len(joined)} rows. Missing:\n{missing}")

    # Use the schedule the sportsbook line covered, never the number of games a
    # team happened to complete. The 2020-21 market was explicitly 72 games;
    # 2019-20 lines covered 82 games before the season was interrupted.
    joined["scheduled_games_for_line"] = joined["season_end_year"].map(
        scheduled_games_for_line
    )
    joined["predicted_wins_82"] = joined["predicted_win_pct"] * STANDARD_SEASON_GAMES
    joined["predicted_wins"] = (
        joined["predicted_win_pct"] * joined["scheduled_games_for_line"]
    )
    joined["predicted_wins_for_line"] = joined["predicted_wins"]
    joined["pick"] = np.where(
        joined["predicted_wins"] > joined["win_total_line"], "over", "under"
    )
    joined["model_edge_wins"] = joined["predicted_wins"] - joined["win_total_line"]

    # Forecast comparison uses an 82-game pace for the interrupted 2019-20 season.
    joined["comparison_actual_wins"] = np.where(
        joined["season_end_year"].eq(COVID_VOID_END_YEAR),
        joined["win_pct"] * STANDARD_SEASON_GAMES,
        joined["wins"],
    )
    joined["model_abs_error_wins"] = (
        joined["predicted_wins"] - joined["comparison_actual_wins"]
    ).abs()
    joined["market_abs_error_wins"] = (
        joined["win_total_line"] - joined["comparison_actual_wins"]
    ).abs()

    joined["grade"] = [
        "void"
        if year == COVID_VOID_END_YEAR
        else grade_pick(pick, wins, line)
        for year, pick, wins, line in zip(
            joined["season_end_year"], joined["pick"], joined["wins"], joined["win_total_line"]
        )
    ]
    win_profit = american_win_profit(args.stake, args.american_odds)
    joined["stake"] = args.stake
    joined["profit"] = joined["grade"].map(
        {"win": win_profit, "loss": -args.stake, "push": 0.0, "void": 0.0}
    )
    joined["cash_return"] = args.stake + joined["profit"]

    summary_rows = []
    for (season, end_year), group in joined.groupby(["season", "season_end_year"], sort=True):
        settled = group["grade"].isin(["win", "loss"])
        model_errors = group["predicted_wins"] - group["comparison_actual_wins"]
        market_errors = group["win_total_line"] - group["comparison_actual_wins"]
        summary_rows.append(
            {
                "season": season,
                "season_end_year": end_year,
                "bets": len(group),
                "wins": int((group["grade"] == "win").sum()),
                "losses": int((group["grade"] == "loss").sum()),
                "pushes": int((group["grade"] == "push").sum()),
                "voids": int((group["grade"] == "void").sum()),
                "win_rate_settled": float((group.loc[settled, "grade"] == "win").mean()) if settled.any() else np.nan,
                "amount_staked": float(group["stake"].sum()),
                "settled_stake": float(settled.sum() * args.stake),
                "net_profit": float(group["profit"].sum()),
                "roi_on_settled_stake": float(group["profit"].sum() / (settled.sum() * args.stake)) if settled.any() else np.nan,
                "model_mae_wins": float(np.abs(model_errors).mean()),
                "market_mae_wins": float(np.abs(market_errors).mean()),
                "model_rmse_wins": float(np.sqrt(np.mean(model_errors**2))),
                "market_rmse_wins": float(np.sqrt(np.mean(market_errors**2))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["cumulative_profit"] = summary["net_profit"].cumsum()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined.sort_values(["season_end_year", "team"]).to_csv(
        output_dir / "all_bets.csv", index=False
    )
    summary.to_csv(output_dir / "yearly_summary.csv", index=False)
    assumptions = {
        "model": args.model,
        "walk_forward": True,
        "burn_in_seasons": [2014, 2015, 2016],
        "bet_seasons": list(BET_END_YEARS),
        "stake_per_team": args.stake,
        "assumed_american_odds": args.american_odds,
        "win_profit_per_10_at_minus_110": american_win_profit(10.0, -110),
        "covid_2019_20": "All bets voided because the scheduled season was shortened; stakes returned.",
        "shortened_2020_21": (
            "Model win percentage is converted to the 72-game schedule before comparison "
            "with 2020-21 preseason lines and final win totals."
        ),
        "selection_rule": "Over if predicted wins > line; Under otherwise.",
        "line_source": "Basketball-Reference preseason odds tables, credited there to SportsOddsHistory.com",
    }
    (output_dir / "assumptions.json").write_text(json.dumps(assumptions, indent=2) + "\n")

    settled = joined["grade"].isin(["win", "loss"])
    wins = int((joined["grade"] == "win").sum())
    losses = int((joined["grade"] == "loss").sum())
    settled_bets = int(settled.sum())
    interval_low, interval_high = wilson_interval(wins, settled_bets)
    breakeven_probability = abs(args.american_odds) / (abs(args.american_odds) + 100)
    overall = {
        "bets": len(joined),
        "settled_bets": settled_bets,
        "wins": wins,
        "losses": losses,
        "pushes": int((joined["grade"] == "push").sum()),
        "voids": int((joined["grade"] == "void").sum()),
        "win_rate": wins / settled_bets,
        "win_rate_wilson_95_low": interval_low,
        "win_rate_wilson_95_high": interval_high,
        "assumed_breakeven_win_rate": breakeven_probability,
        "one_sided_binomial_p_vs_assumed_breakeven": binomial_upper_tail(
            wins, settled_bets, breakeven_probability
        ),
        "amount_staked": float(joined["stake"].sum()),
        "settled_stake": float(settled.sum() * args.stake),
        "net_profit": float(joined["profit"].sum()),
        "roi_on_settled_stake": float(joined["profit"].sum() / (settled.sum() * args.stake)),
        "overall_model_mae_wins": float(joined["model_abs_error_wins"].mean()),
        "overall_market_mae_wins": float(joined["market_abs_error_wins"].mean()),
    }
    (output_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2) + "\n")

    sensitivity_rows = []
    for price in (-105, -110, -115, -120):
        profit = wins * american_win_profit(args.stake, price) - losses * args.stake
        sensitivity_rows.append(
            {
                "american_odds": price,
                "net_profit": profit,
                "roi_on_settled_stake": profit / (settled_bets * args.stake),
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(output_dir / "odds_sensitivity.csv", index=False)

    print(summary.to_string(index=False))
    print("\nTOTAL")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
