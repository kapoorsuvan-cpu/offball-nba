#!/usr/bin/env python3
"""Build the verified 2026-27 roster snapshot and current model predictions.

ESPN is the roster/position/headshot authority. Current NBA 2K27 ratings are
merged by player name from the browser-collected 2KRatings team tables. If a
current rating is missing, the script falls back to the player's 2025-26 CSV
rating and finally to a clearly labeled unrated estimate for new players.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import Pipeline

from train_models import features, model_specs, preprocessor


ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams?limit=50"
)
ESPN_ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/"
    "{team_id}/roster?season=2027"
)
NBA_OFFSEASON_URL = "https://www.nba.com/news/nba-offseason-deals-2026"
NBA_TRADE_TRACKER_URL = "https://www.nba.com/news/2026-offseason-trade-tracker"

NAME_ALIASES = {
    "aj johnson": "a j johnson",
    "kj martin": "kenyon martin",
    "pj washington": "p j washington",
    "rj barrett": "r j barrett",
    "ron harper": "ron harper",
    "taze moore": "tazé moore",
}

RATING_TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
}


def normalize_name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    ascii_name = ascii_name.lower().replace("&", "and")
    ascii_name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", ascii_name)
    ascii_name = re.sub(r"[^a-z0-9]+", " ", ascii_name).strip()
    return NAME_ALIASES.get(ascii_name, ascii_name)


def canonical_position(position: str | None) -> str:
    value = (position or "").upper().strip()
    aliases = {
        "G": "PG",
        "F": "SF",
        "G/F": "SG",
        "F/G": "SF",
        "F/C": "PF",
        "C/F": "C",
        "FORWARD": "SF",
        "GUARD": "PG",
        "CENTER": "C",
    }
    return aliases.get(value, value if value in {"PG", "SG", "SF", "PF", "C"} else "SF")


def fetch_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=30)
    if response.ok:
        return response.json()
    # ESPN currently rejects Python's TLS fingerprint while serving the same
    # public JSON to browsers and curl. Keep a deterministic, argument-safe
    # curl fallback instead of weakening verification or changing providers.
    result = subprocess.run(
        ["curl", "-fsSL", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def latest_historical_ratings(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    latest = frame.loc[frame["season_end_year"] == frame["season_end_year"].max()]
    output: dict[str, dict[str, Any]] = {}
    for _, row in latest.iterrows():
        for rank in range(1, 11):
            player = row.get(f"player_{rank}")
            rating = row.get(f"rating_{rank}")
            if pd.isna(player) or pd.isna(rating):
                continue
            output[normalize_name(str(player))] = {
                "rating": int(rating),
                "position": canonical_position(str(row.get(f"position_{rank}", ""))),
                "season": str(row["season"]),
            }
    return output


def make_rating_indexes(ratings: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    global_index: dict[str, Any] = {}
    team_index: dict[str, Any] = {}
    for team in ratings:
        team_key = team["team"]
        team_index[team_key] = {}
        for player in team["players"]:
            key = normalize_name(player["player"])
            item = {**player, "listed_team": team_key, "team_source_url": team["source_url"]}
            team_index[team_key][key] = item
            global_index[key] = item
    return global_index, team_index


def build_feature_row(players: list[dict[str, Any]]) -> dict[str, Any]:
    top = sorted(players, key=lambda player: (-player["rating"], player["name"]))[:10]
    if not top:
        raise ValueError("Cannot build prediction features without players")
    while len(top) < 10:
        top.append({"rating": 68, "position": "SF"})
    row: dict[str, Any] = {}
    for rank, player in enumerate(top, start=1):
        row[f"rating_{rank}"] = float(player["rating"])
        row[f"position_{rank}"] = canonical_position(player["position"])
    return row


def predict_wins(model: Pipeline, row: dict[str, Any]) -> float:
    frame = pd.DataFrame([row])
    return float(np.clip(model.predict(frame)[0] * 82.0, 8.0, 74.0))


def normal_distribution(mean: float, sigma: float) -> list[dict[str, float]]:
    buckets = list(range(20, 81, 5))
    values = [math.exp(-0.5 * ((bucket - mean) / sigma) ** 2) for bucket in buckets]
    total = sum(values)
    return [
        {"wins": bucket, "probability": round(value / total, 4)}
        for bucket, value in zip(buckets, values, strict=True)
    ]


def model_metrics(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    output = []
    labels = {
        "extra_trees": "Extra Trees",
        "elastic_net": "Elastic Net",
        "tabfm_jax": "TabFM",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }
    for _, row in frame.iterrows():
        name = str(row["model"])
        if name not in labels:
            continue
        output.append(
            {
                "id": name,
                "name": labels[name],
                "maeWins": round(float(row["mae"]) * 82, 2),
                "rmseWins": round(float(row["rmse"]) * 82, 2),
                "r2": round(float(row["r2"]), 3),
                "cvRmseWins": (
                    round(float(row["cv_rmse"]) * 82, 2)
                    if pd.notna(row["cv_rmse"])
                    else None
                ),
            }
        )
    return sorted(output, key=lambda item: item["rmseWins"])


def backtest_summary(yearly_path: Path, overall_path: Path) -> dict[str, Any]:
    yearly = pd.read_csv(yearly_path)
    overall = json.loads(overall_path.read_text())
    years = []
    for _, row in yearly.iterrows():
        years.append(
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
        )
    return {
        "years": years,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="data/2k27_current_ratings.browser.json")
    parser.add_argument("--history", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--metrics", default="artifacts/model_metrics.csv")
    parser.add_argument(
        "--yearly-backtest",
        default="artifacts/win_total_backtest_walkforward_cv/yearly_summary.csv",
    )
    parser.add_argument(
        "--overall-backtest",
        default="artifacts/win_total_backtest_walkforward_cv/overall_summary.json",
    )
    parser.add_argument("--output", default="dashboard-app/app/data/current-rosters.json")
    args = parser.parse_args()

    ratings = json.loads(Path(args.ratings).read_text())
    history = pd.read_csv(args.history)
    global_ratings, team_ratings = make_rating_indexes(ratings)
    fallback_ratings = latest_historical_ratings(history)

    X, rating_columns, position_columns = features(history)
    model = Pipeline(
        [
            ("prep", preprocessor(rating_columns, position_columns)),
            ("model", model_specs()["extra_trees"]),
        ]
    )
    model.fit(X, history["win_pct"].to_numpy(dtype=float))

    metrics = model_metrics(Path(args.metrics))
    extra_trees_metrics = next(item for item in metrics if item["id"] == "extra_trees")
    sigma_wins = max(7.0, extra_trees_metrics["rmseWins"])
    confidence_delta = 1.282 * sigma_wins

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": "https://www.espn.com/",
        }
    )
    teams_payload = fetch_json(session, ESPN_TEAMS_URL)
    espn_teams = teams_payload["sports"][0]["leagues"][0]["teams"]
    if len(espn_teams) != 30:
        raise ValueError(f"Expected 30 ESPN teams, received {len(espn_teams)}")

    teams: list[dict[str, Any]] = []
    audit = Counter()
    cross_team_matches: list[dict[str, str]] = []
    unmatched_players: list[dict[str, str]] = []

    for wrapper in espn_teams:
        team_meta = wrapper["team"]
        roster_payload = fetch_json(session, ESPN_ROSTER_URL.format(team_id=team_meta["id"]))
        espn_team_name = roster_payload["team"]["displayName"]
        rating_team_name = RATING_TEAM_ALIASES.get(espn_team_name, espn_team_name)
        same_team_index = team_ratings.get(rating_team_name, {})
        players: list[dict[str, Any]] = []

        for athlete in roster_payload.get("athletes", []):
            name = athlete["fullName"]
            key = normalize_name(name)
            current = same_team_index.get(key)
            rating_source = "NBA 2K27 current"
            if current is not None:
                audit["same_team_2k27"] += 1
            else:
                current = global_ratings.get(key)
                if current is not None:
                    audit["cross_team_2k27"] += 1
                    cross_team_matches.append(
                        {
                            "player": name,
                            "espnTeam": espn_team_name,
                            "ratingPageTeam": current["listed_team"],
                        }
                    )
                elif key in fallback_ratings:
                    current = fallback_ratings[key]
                    rating_source = "NBA 2K26 fallback"
                    audit["2k26_fallback"] += 1
                else:
                    current = {"rating": 68, "position": None}
                    rating_source = "Unrated estimate"
                    audit["unrated_estimate"] += 1
                    unmatched_players.append({"player": name, "team": espn_team_name})

            raw_espn_position = (athlete.get("position") or {}).get("abbreviation", "")
            raw_rating_positions = current.get("positions") or [current.get("position")]
            rating_positions = [
                canonical_position(value)
                for value in raw_rating_positions
                if value
            ]
            if rating_positions:
                positions = list(dict.fromkeys(rating_positions))
                position_source = (
                    "NBA 2K27 current"
                    if rating_source == "NBA 2K27 current"
                    else "NBA 2K26 fallback"
                )
            else:
                positions = [canonical_position(raw_espn_position)]
                position_source = "ESPN roster fallback"
            position = positions[0]
            player_page = next(
                (
                    link["href"]
                    for link in athlete.get("links", [])
                    if "playercard" in link.get("rel", [])
                ),
                None,
            )
            published_headshot = (athlete.get("headshot") or {}).get("href")
            players.append(
                {
                    "id": str(athlete["id"]),
                    "name": name,
                    "jersey": athlete.get("jersey"),
                    "position": position,
                    "positions": positions,
                    "positionDisplay": "/".join(positions),
                    "positionSource": position_source,
                    "rating": int(current["rating"]),
                    "ratingSource": rating_source,
                    "ratingSourceUrl": current.get("team_source_url") or current.get("profile_url"),
                    "ratingPageTeam": current.get("listed_team"),
                    "headshotUrl": published_headshot
                    or f"https://a.espncdn.com/i/headshots/nba/players/full/{athlete['id']}.png",
                    "headshotVerified": bool(published_headshot),
                    "espnUrl": player_page,
                    "age": athlete.get("age"),
                    "height": athlete.get("displayHeight"),
                    "weight": athlete.get("displayWeight"),
                    "status": (athlete.get("status") or {}).get("name", "Active"),
                    "experienceYears": (athlete.get("experience") or {}).get("years"),
                }
            )

        if len(players) < 10:
            raise ValueError(f"{espn_team_name} roster has only {len(players)} players")
        players.sort(key=lambda player: (-player["rating"], player["name"]))
        feature_row = build_feature_row(players)
        prediction = predict_wins(model, feature_row)

        sensitivities: list[float] = []
        for rank in range(1, 11):
            low = dict(feature_row)
            high = dict(feature_row)
            low[f"rating_{rank}"] = max(40.0, low[f"rating_{rank}"] - 3.0)
            high[f"rating_{rank}"] = min(99.0, high[f"rating_{rank}"] + 3.0)
            sensitivity = (predict_wins(model, high) - predict_wins(model, low)) / 6.0
            sensitivities.append(round(sensitivity, 3))

        for rank, player in enumerate(players):
            sensitivity = sensitivities[rank] if rank < 10 else 0.0
            player["modelRank"] = rank + 1 if rank < 10 else None
            player["ratingSensitivity"] = sensitivity
            # A compact, comparable local scenario: the win change from a
            # three-point rating movement near the current roster state.
            player["impactWins"] = round(abs(sensitivity) * 3.0, 1)

        position_mix = Counter(player["position"] for player in players[:10])
        teams.append(
            {
                "id": str(team_meta["id"]),
                "name": espn_team_name,
                "shortName": roster_payload["team"]["name"],
                "abbreviation": roster_payload["team"]["abbreviation"],
                "slug": team_meta["slug"],
                "logoUrl": roster_payload["team"]["logo"],
                "color": f"#{roster_payload['team'].get('color') or team_meta.get('color') or '8ad30a'}",
                "espnRosterUrl": next(
                    (
                        link["href"]
                        for link in team_meta.get("links", [])
                        if "roster" in link.get("rel", [])
                    ),
                    roster_payload["team"]["clubhouse"],
                ),
                "rosterCount": len(players),
                "players": players,
                "topTen": [player["id"] for player in players[:10]],
                "positionMix": {position: position_mix.get(position, 0) for position in ["PG", "SG", "SF", "PF", "C"]},
                "prediction": {
                    "wins": round(prediction, 1),
                    "losses": round(82 - prediction, 1),
                    "record": f"{round(prediction)}-{82 - round(prediction)}",
                    "confidenceLow": max(0, round(prediction - confidence_delta)),
                    "confidenceHigh": min(82, round(prediction + confidence_delta)),
                    "distribution": normal_distribution(prediction, sigma_wins),
                    "model": "Extra Trees",
                },
            }
        )

    teams.sort(key=lambda team: (-team["prediction"]["wins"], team["name"]))

    # Explicit transaction assertions backed by the NBA's 2026 trackers.
    team_by_player = {
        normalize_name(player["name"]): team["name"]
        for team in teams
        for player in team["players"]
    }
    expected_moves = {
        "Ja Morant": "Portland Trail Blazers",
        "Luguentz Dort": "Atlanta Hawks",
        "Paul George": "Boston Celtics",
        "Mike Conley": "Boston Celtics",
        "Mitchell Robinson": "Boston Celtics",
        "Jaylen Brown": "Philadelphia 76ers",
        "LeBron James": "Philadelphia 76ers",
    }
    move_checks = []
    for player, expected_team in expected_moves.items():
        actual_team = team_by_player.get(normalize_name(player))
        if actual_team != expected_team:
            raise ValueError(f"Move check failed: {player} expected {expected_team}, got {actual_team}")
        move_checks.append({"player": player, "team": actual_team, "status": "verified"})

    output = {
        "metadata": {
            "season": "2026-27",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "generatedLocalDate": datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat(),
            "rosterAuthority": "ESPN 2026-27 roster API",
            "headshotAuthority": "ESPN CDN",
            "ratingAuthority": "2KRatings NBA 2K27 current team tables",
            "fallbackPolicy": "2025-26 CSV rating, then 68 only for players with no prior rating",
            "model": "Extra Trees trained on 2013-14 through 2025-26 team seasons",
            "sources": [
                {"label": "ESPN NBA teams and rosters", "url": ESPN_TEAMS_URL},
                {"label": "NBA 2026 offseason deals tracker", "url": NBA_OFFSEASON_URL},
                {"label": "NBA 2026 trade tracker", "url": NBA_TRADE_TRACKER_URL},
                {"label": "Current 2K27 ratings", "url": "https://www.2kratings.com/teams"},
            ],
        },
        "audit": {
            "teamCount": len(teams),
            "playerCount": sum(team["rosterCount"] for team in teams),
            "headshotUrlCount": sum(
                1 for team in teams for player in team["players"] if player["headshotUrl"]
            ),
            "verifiedHeadshotCount": sum(
                1 for team in teams for player in team["players"] if player["headshotVerified"]
            ),
            "matchCounts": dict(audit),
            "positionSourceCounts": dict(
                Counter(
                    player["positionSource"]
                    for team in teams
                    for player in team["players"]
                )
            ),
            "invalidPositionPlayers": [
                {"player": player["name"], "team": team["name"], "positions": player["positions"]}
                for team in teams
                for player in team["players"]
                if not player["positions"]
                or any(position not in {"PG", "SG", "SF", "PF", "C"} for position in player["positions"])
            ],
            "crossTeamRatingMatches": cross_team_matches,
            "unmatchedPlayers": unmatched_players,
            "officialMoveChecks": move_checks,
        },
        "modelMetrics": metrics,
        "backtest": backtest_summary(Path(args.yearly_backtest), Path(args.overall_backtest)),
        "teams": teams,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "teams": output["audit"]["teamCount"],
                "players": output["audit"]["playerCount"],
                "headshotUrls": output["audit"]["headshotUrlCount"],
                "verifiedHeadshots": output["audit"]["verifiedHeadshotCount"],
                "matches": output["audit"]["matchCounts"],
                "crossTeamMatches": len(cross_team_matches),
                "unmatched": unmatched_players,
                "moveChecks": move_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
