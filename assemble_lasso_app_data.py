#!/usr/bin/env python3
"""Assemble current and historical dashboard data for the selected Lasso model."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from retest_lasso_models import engineer_features


CURRENT_END_YEAR = 2027
STANDARD_SEASON_GAMES = 82
PREDICTION_SCHEDULE_GAMES = {2021: 72}
TEAM_ALIASES = {"LA Clippers": "Los Angeles Clippers"}
REVERSE_TEAM_ALIASES = {value: key for key, value in TEAM_ALIASES.items()}


def normalize_name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).strip()


def historical_positions(value: Any) -> list[str]:
    raw = str(value).upper().strip()
    mapping = {
        "G": ["PG", "SG"],
        "F": ["SF", "PF"],
        "G/F": ["SG", "SF"],
        "F/G": ["SF", "SG"],
        "F/C": ["PF", "C"],
        "C/F": ["C", "PF"],
    }
    if raw in {"PG", "SG", "SF", "PF", "C"}:
        return [raw]
    return mapping.get(raw, ["SF"])


def normal_distribution(
    mean: float,
    sigma: float,
    schedule_games: int = STANDARD_SEASON_GAMES,
) -> list[dict[str, float]]:
    buckets = list(range(10 if schedule_games < STANDARD_SEASON_GAMES else 15, schedule_games, 5))
    values = [math.exp(-0.5 * ((bucket - mean) / sigma) ** 2) for bucket in buckets]
    total = sum(values)
    return [
        {"wins": bucket, "probability": round(value / total, 4)}
        for bucket, value in zip(buckets, values, strict=True)
    ]


def prior_wins(history: pd.DataFrame, team_name: str, count: int = 3) -> list[float]:
    historical_name = TEAM_ALIASES.get(team_name, team_name)
    rows = history.loc[history["team"].eq(historical_name)].sort_values(
        "season_end_year", ascending=False
    )
    values = (rows["win_pct"] * 82.0).tolist()
    return [float(values[index]) if index < len(values) else np.nan for index in range(count)]


def current_raw_rows(data: dict[str, Any], history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team in data["teams"]:
        top = sorted(team["players"], key=lambda player: (-player["rating"], player["name"]))[:10]
        row: dict[str, Any] = {
            "team": TEAM_ALIASES.get(team["name"], team["name"]),
            "season_end_year": CURRENT_END_YEAR,
        }
        for rank, player in enumerate(top, start=1):
            row[f"rating_{rank}"] = float(player["rating"])
            row[f"position_{rank}"] = player["position"]
        for lag, wins in enumerate(prior_wins(history, team["name"]), start=1):
            row[f"prior_wins_{lag}"] = wins
        rows.append(row)
    return pd.DataFrame(rows)


def predict_all(model: Any, raw: pd.DataFrame) -> np.ndarray:
    engineered = engineer_features(raw)
    if isinstance(model, dict):
        pipeline = model["pipeline"]
        features = model["features"]
        base = np.asarray(pipeline.predict(engineered[features]), dtype=float)
        curve = np.asarray(model["rankCurve"], dtype=float)
        ranks = np.argsort(np.argsort(-base))
        blend = float(model["rankBlend"])
        predictions = (1.0 - blend) * base + blend * curve[ranks]
    else:
        predictions = model.predict(engineered)
    return np.clip(np.asarray(predictions, dtype=float), 8.0, 74.0)


def rating_sensitivity(
    model: Any,
    raw: pd.DataFrame,
    team_index: int,
    ranks: list[int],
) -> float:
    low = raw.copy()
    high = raw.copy()
    for rank in ranks:
        column = f"rating_{rank}"
        low.loc[team_index, column] = max(40.0, float(low.loc[team_index, column]) - 3.0)
        high.loc[team_index, column] = min(99.0, float(high.loc[team_index, column]) + 3.0)
    return float((predict_all(model, high)[team_index] - predict_all(model, low)[team_index]) / 6.0)


def model_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    metrics = pd.read_csv(metrics_path)
    names = {
        "rank_calibrated_lasso": "Spread-Calibrated Lasso",
        "elastic_net": "Elastic Net",
        "extra_trees": "Extra Trees",
        "tabfm_jax": "TabFM",
    }
    output = []
    for _, row in metrics.iterrows():
        output.append(
            {
                "id": str(row["model"]),
                "name": names.get(str(row["model"]), str(row["model"])),
                "maeWins": round(float(row["maeWins"]), 2),
                "rmseWins": round(float(row["rmseWins"]), 2),
                "r2": round(float(row["r2"]), 3),
                "cvRmseWins": (
                    round(float(row["validationRmseWins"]), 2)
                    if pd.notna(row["validationRmseWins"])
                    else None
                ),
            }
        )
    return output


def backtest_summary(yearly_path: Path, overall_path: Path) -> dict[str, Any]:
    yearly = pd.read_csv(yearly_path)
    overall = json.loads(overall_path.read_text())
    return {
        "years": [
            {
                "season": str(row["season"]),
                "bets": int(row["bets"]),
                "wins": int(row["wins"]),
                "losses": int(row["losses"]),
                "voids": int(row["voids"]),
                "netProfit": round(float(row["net_profit"]), 2),
                "modelMaeWins": round(float(row["model_mae_wins"]), 2),
                "marketMaeWins": round(float(row["market_mae_wins"]), 2),
            }
            for _, row in yearly.iterrows()
        ],
        "overall": {
            "bets": int(overall["bets"]),
            "settledBets": int(overall["settled_bets"]),
            "wins": int(overall["wins"]),
            "losses": int(overall["losses"]),
            "voids": int(overall["voids"]),
            "winRate": round(float(overall["win_rate"]), 4),
            "netProfit": round(float(overall["net_profit"]), 2),
            "roi": round(float(overall["roi_on_settled_stake"]), 4),
            "modelMaeWins": round(float(overall["overall_model_mae_wins"]), 2),
            "marketMaeWins": round(float(overall["overall_market_mae_wins"]), 2),
            "pValue": round(float(overall["one_sided_binomial_p_vs_assumed_breakeven"]), 3),
        },
    }


def update_current_teams(
    data: dict[str, Any],
    history: pd.DataFrame,
    model: Any,
    sigma_wins: float,
) -> None:
    raw = current_raw_rows(data, history)
    predictions = predict_all(model, raw)
    confidence_delta = 1.282 * sigma_wins
    for team_index, team in enumerate(data["teams"]):
        players = sorted(team["players"], key=lambda player: (-player["rating"], player["name"]))
        team["players"] = players
        team["topTen"] = [player["id"] for player in players[:10]]
        individual = [
            rating_sensitivity(model, raw, team_index, [rank]) for rank in range(1, 11)
        ]
        starter_core = rating_sensitivity(model, raw, team_index, [3, 4, 5])
        bench_depth = rating_sensitivity(model, raw, team_index, [6, 7, 8, 9, 10])
        for rank, player in enumerate(players):
            sensitivity = individual[rank] if rank < 10 else 0.0
            player["modelRank"] = rank + 1 if rank < 10 else None
            player["ratingSensitivity"] = round(sensitivity, 3)
            player["impactWins"] = round(abs(sensitivity) * 3.0, 1)
        prediction = float(predictions[team_index])
        team["prediction"] = {
            "wins": round(prediction, 1),
            "losses": round(82.0 - prediction, 1),
            "record": f"{round(prediction)}-{82 - round(prediction)}",
            "scheduleGames": STANDARD_SEASON_GAMES,
            "confidenceLow": max(0, round(prediction - confidence_delta)),
            "confidenceHigh": min(82, round(prediction + confidence_delta)),
            "distribution": normal_distribution(prediction, sigma_wins),
            "model": "Spread-Calibrated Lasso",
            "scenarioSensitivities": {
                "starOne": round(individual[0], 3),
                "starTwo": round(individual[1], 3),
                "starterCore": round(starter_core, 3),
                "benchDepth": round(bench_depth, 3),
            },
        }


def historical_seasons(
    data: dict[str, Any],
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    historical_metadata: dict[tuple[int, str, str], dict[str, Any]],
    sigma_wins: float,
) -> list[dict[str, Any]]:
    current_team_by_history_name = {
        TEAM_ALIASES.get(team["name"], team["name"]): team for team in data["teams"]
    }
    current_player_by_name = {
        normalize_name(player["name"]): player
        for team in data["teams"]
        for player in team["players"]
    }
    prediction_index = forecast.set_index(["season_end_year", "team"])["predicted_wins"]
    seasons: list[dict[str, Any]] = []
    for end_year in range(2017, 2027):
        schedule_games = PREDICTION_SCHEDULE_GAMES.get(end_year, STANDARD_SEASON_GAMES)
        schedule_scale = schedule_games / STANDARD_SEASON_GAMES
        scheduled_sigma_wins = sigma_wins * schedule_scale
        confidence_delta = 1.282 * scheduled_sigma_wins
        season_frame = history.loc[history["season_end_year"].eq(end_year)]
        teams: list[dict[str, Any]] = []
        for _, row in season_frame.iterrows():
            current_team = current_team_by_history_name[str(row["team"])]
            players: list[dict[str, Any]] = []
            for rank in range(1, 11):
                name = str(row[f"player_{rank}"])
                rating = int(row[f"rating_{rank}"])
                player_metadata = historical_metadata.get(
                    (end_year, str(row["team"]), normalize_name(name))
                )
                positions = (
                    [str(player_metadata["position"])]
                    if player_metadata
                    else historical_positions(row[f"position_{rank}"])
                )
                current_player = current_player_by_name.get(normalize_name(name))
                headshot_url = (
                    player_metadata.get("headshotUrl") if player_metadata else None
                ) or (current_player.get("headshotUrl") if current_player else None)
                espn_url = (
                    player_metadata.get("espnUrl") if player_metadata else None
                ) or (current_player.get("espnUrl") if current_player else None)
                players.append(
                    {
                        "id": f"hist-{end_year}-{normalize_name(name).replace(' ', '-')}",
                        "name": name,
                        "jersey": None,
                        "position": positions[0],
                        "positions": positions,
                        "positionDisplay": "/".join(positions),
                        "positionSource": (
                            str(player_metadata["positionSource"])
                            if player_metadata
                            else str(row[f"position_source_{rank}"])
                        ),
                        "rating": rating,
                        "ratingSource": f"NBA 2K{str(end_year)[-2:]}",
                        "ratingSourceUrl": "https://hoopshype.com/nba2k/",
                        "ratingPageTeam": str(row["team"]),
                        "headshotUrl": headshot_url,
                        "headshotVerified": bool(
                            (player_metadata and player_metadata.get("headshotVerified"))
                            or (current_player and current_player.get("headshotVerified"))
                        ),
                        "espnUrl": espn_url,
                        "age": None,
                        "height": None,
                        "weight": None,
                        "status": "Historical roster",
                        "experienceYears": None,
                        "modelRank": rank,
                        "ratingSensitivity": 0.0,
                        "impactWins": 0.0,
                    }
                )
            prediction_82 = float(prediction_index.loc[(end_year, row["team"])])
            prediction = prediction_82 * schedule_scale
            games_played = int(row["wins"] + row["losses"])
            normalized_actual_wins = float(row["win_pct"] * 82.0)
            position_mix = Counter(player["position"] for player in players)
            teams.append(
                {
                    "id": f"{end_year}-{current_team['id']}",
                    "name": str(row["team"]),
                    "shortName": current_team["shortName"],
                    "abbreviation": current_team["abbreviation"],
                    "slug": current_team["slug"],
                    "logoUrl": current_team["logoUrl"],
                    "color": current_team["color"],
                    "espnRosterUrl": current_team["espnRosterUrl"],
                    "rosterCount": 10,
                    "players": players,
                    "topTen": [player["id"] for player in players],
                    "positionMix": {
                        position: position_mix.get(position, 0)
                        for position in ["PG", "SG", "SF", "PF", "C"]
                    },
                    "prediction": {
                        "wins": round(prediction, 1),
                        "losses": round(schedule_games - prediction, 1),
                        "record": f"{round(prediction)}-{schedule_games - round(prediction)}",
                        "scheduleGames": schedule_games,
                        "confidenceLow": max(0, round(prediction - confidence_delta)),
                        "confidenceHigh": min(
                            schedule_games, round(prediction + confidence_delta)
                        ),
                        "distribution": normal_distribution(
                            prediction, scheduled_sigma_wins, schedule_games
                        ),
                        "model": "Spread-Calibrated Lasso walk-forward",
                        "scenarioSensitivities": None,
                    },
                    "actual": {
                        "wins": int(row["wins"]),
                        "losses": int(row["losses"]),
                        "record": f"{int(row['wins'])}-{int(row['losses'])}",
                        "gamesPlayed": games_played,
                        "winPct": round(float(row["win_pct"]), 6),
                        "normalizedWins": round(normalized_actual_wins, 1),
                    },
                }
            )
        seasons.append(
            {
                "season": str(season_frame.iloc[0]["season"]),
                "seasonEndYear": end_year,
                "teams": sorted(teams, key=lambda team: team["name"]),
            }
        )
    return seasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dashboard-app/app/data/current-rosters.json")
    parser.add_argument("--output", default="dashboard-app/app/data/current-rosters.json")
    parser.add_argument("--history", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--model", default="artifacts/rank_calibrated_lasso/selected_model.joblib")
    parser.add_argument("--summary", default="artifacts/rank_calibrated_lasso/summary.json")
    parser.add_argument("--metrics", default="artifacts/rank_calibrated_lasso/model_metrics.csv")
    parser.add_argument("--forecast", default="artifacts/rank_lasso_walkforward_backtest/walkforward_predictions.csv")
    parser.add_argument(
        "--historical-metadata",
        default="data/historical_player_metadata.json",
    )
    parser.add_argument("--yearly", default="artifacts/rank_lasso_walkforward_backtest/yearly_summary.csv")
    parser.add_argument("--overall", default="artifacts/rank_lasso_walkforward_backtest/overall_summary.json")
    parser.add_argument(
        "--previous-freeze-manifest",
        default="artifacts/prediction_freezes/2026-27-as-of-2026-08-10.manifest.json",
    )
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    history = pd.read_csv(args.history)
    model = joblib.load(args.model)
    summary = json.loads(Path(args.summary).read_text())
    metrics = model_metrics(Path(args.metrics))
    selected_metric = next(
        metric for metric in metrics if metric["id"] == "rank_calibrated_lasso"
    )
    sigma_wins = float(selected_metric["rmseWins"])
    previous_freeze_path = Path(args.previous_freeze_manifest)
    previous_range = (
        float(json.loads(previous_freeze_path.read_text())["predictionRangeWins"])
        if previous_freeze_path.exists()
        else None
    )

    update_current_teams(data, history, model, sigma_wins)
    forecast = pd.read_csv(args.forecast)
    metadata_payload = json.loads(Path(args.historical_metadata).read_text())
    historical_metadata = {
        (
            int(player["seasonEndYear"]),
            str(player["team"]),
            normalize_name(str(player["player"])),
        ): player
        for player in metadata_payload["players"]
    }
    data["historicalSeasons"] = historical_seasons(
        data, history, forecast, historical_metadata, sigma_wins
    )
    data["historicalMetadataAudit"] = metadata_payload["metadata"]
    data["modelMetrics"] = metrics
    data["backtest"] = backtest_summary(Path(args.yearly), Path(args.overall))
    data["metadata"].update(
        {
            "model": "Starter-weighted Lasso with league-spread calibration, trained on 2013-14 through 2025-26",
            "modelSelection": summary["selectionRule"],
            "selectedFeatureCount": int(summary["retainedFeatureCount"]),
            "selectedFeatures": [
                {
                    "feature": feature["feature"],
                    "coefficient_wins": round(float(feature["coefficientWins"]), 4),
                }
                for feature in summary["features"]
                if feature["retained"]
            ],
            "validationMaeWins": round(float(summary["expandingValidation"]["maeWins"]), 2),
            "validationRmseWins": round(float(summary["expandingValidation"]["rmseWins"]), 2),
            "heldoutMaeWins": selected_metric["maeWins"],
            "heldoutRmseWins": selected_metric["rmseWins"],
            "heldoutR2": selected_metric["r2"],
            "currentPredictionRangeWins": round(float(summary["currentPredictionRangeWins"]), 2),
            "previousPredictionRangeWins": (
                round(previous_range, 2) if previous_range is not None else None
            ),
            "rankCalibrationBlend": round(float(summary["rankBlend"]), 3),
            "minimumValidationSpreadRatio": round(float(summary["minimumValidationSpreadRatio"]), 3),
            "validationSpreadRatio": round(float(summary["expandingValidation"]["spreadRatio"]), 3),
            "heldoutSpreadRatio": round(float(summary["holdout"]["spreadRatio"]), 3),
            "validationTailBias": {
                key: round(float(value), 2)
                for key, value in summary["validationTailBias"].items()
            },
            "heldoutTailBias": {
                key: round(float(value), 2)
                for key, value in summary["holdoutTailBias"].items()
            },
        }
    )
    data["teams"].sort(key=lambda team: (-team["prediction"]["wins"], team["name"]))

    output = Path(args.output)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "currentTeams": len(data["teams"]),
                "historicalSeasons": len(data["historicalSeasons"]),
                "historicalTeams": sum(len(season["teams"]) for season in data["historicalSeasons"]),
                "model": data["metadata"]["model"],
                "heldoutMaeWins": data["metadata"]["heldoutMaeWins"],
                "currentPredictionRangeWins": data["metadata"]["currentPredictionRangeWins"],
                "backtest": data["backtest"]["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
