#!/usr/bin/env python3
"""Build the compact data bundle used by the OFFBALL scouting dashboard.

The roster snapshot is sourced from the existing OFFBALL data file. Shooting
and matchup results are sourced from shufinskiy/nba_data for the 2025-26
regular season. The requested Kawhi Leonard trade is applied as a roster
override, including the reported Brandon Ingram and Gradey Dick return.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import tarfile
import tempfile
import unicodedata
import urllib.request
from urllib.error import URLError
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_ROSTERS = ROOT / "dashboard-app/app/data/current-rosters.json"
DEFAULT_OUTPUT = ROOT / "dashboard-app/app/scout/data/scouting-data.json"

SHOT_URL = (
    "https://raw.githubusercontent.com/shufinskiy/nba_data/main/"
    "datasets/shotdetail_2025.tar.xz"
)
MATCHUP_URL = (
    "https://raw.githubusercontent.com/shufinskiy/nba_data/main/"
    "datasets/matchups_2025.tar.xz"
)

ZONE_ORDER = (
    ("rim", "Rim"),
    ("paint", "Paint"),
    ("midrange", "Midrange"),
    ("left_corner_3", "Left corner 3"),
    ("right_corner_3", "Right corner 3"),
    ("above_break_3", "Above the break 3"),
)

SHOT_ZONE_MAP = {
    "Restricted Area": "rim",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "midrange",
    "Left Corner 3": "left_corner_3",
    "Right Corner 3": "right_corner_3",
    "Above the Break 3": "above_break_3",
}

# ESPN's roster abbreviations differ from the NBA data files in six places.
NBA_TO_ROSTER_ABBR = {
    "GSW": "GS",
    "NOP": "NO",
    "NYK": "NY",
    "SAS": "SA",
    "UTA": "UTAH",
    "WAS": "WSH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosters", type=Path, default=DEFAULT_ROSTERS)
    parser.add_argument("--shotdetail", type=Path)
    parser.add_argument("--matchups", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalized_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_value.lower())


def number(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def download_csv(url: str, member_name: str) -> io.TextIOWrapper:
    request = urllib.request.Request(url, headers={"User-Agent": "offball-scout/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            archive_bytes = response.read()
    except URLError:
        # Match the existing roster builder: some local Python installations
        # have an incomplete CA bundle even when curl can verify the same URL.
        result = subprocess.run(
            ["curl", "-fsSL", url],
            check=True,
            capture_output=True,
        )
        archive_bytes = result.stdout
    archive = tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:xz")
    member = archive.getmember(member_name)
    extracted = archive.extractfile(member)
    if extracted is None:
        raise RuntimeError(f"Could not read {member_name} from {url}")
    return io.TextIOWrapper(extracted, encoding="utf-8", newline="")


def open_csv(path: Path | None, url: str, member_name: str) -> io.TextIOBase:
    if path:
        return path.open(encoding="utf-8", newline="")
    return download_csv(url, member_name)


def empty_zone_counts() -> dict[str, dict[str, int]]:
    return {key: {"made": 0, "attempts": 0} for key, _ in ZONE_ORDER}


def add_shot(target: dict[str, dict[str, int]], zone: str, made: int) -> None:
    target[zone]["attempts"] += 1
    target[zone]["made"] += made


def percentage(made: float, attempts: float) -> float | None:
    return round(100 * made / attempts, 1) if attempts else None


def zone_rows(
    counts: dict[str, dict[str, int]],
    league_counts: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    qualifying_attempts = sum(item["attempts"] for item in counts.values())
    rows: list[dict[str, Any]] = []
    for key, label in ZONE_ORDER:
        attempts = counts[key]["attempts"]
        made = counts[key]["made"]
        league_attempts = league_counts[key]["attempts"]
        league_made = league_counts[key]["made"]
        pct = percentage(made, attempts)
        league_pct = percentage(league_made, league_attempts)
        rows.append(
            {
                "key": key,
                "label": label,
                "made": made,
                "attempts": attempts,
                "pct": pct,
                "leaguePct": league_pct,
                "delta": round(pct - league_pct, 1)
                if pct is not None and league_pct is not None
                else None,
                "frequency": round(100 * attempts / qualifying_attempts, 1)
                if qualifying_attempts
                else 0,
            }
        )
    return rows


def apply_kawhi_trade(teams: list[dict[str, Any]]) -> None:
    by_abbreviation = {team["abbreviation"]: team for team in teams}
    toronto = by_abbreviation["TOR"]
    clippers = by_abbreviation["LAC"]

    def take(team: dict[str, Any], player_name: str) -> dict[str, Any]:
        for index, player in enumerate(team["players"]):
            if player["name"] == player_name:
                return team["players"].pop(index)
        raise RuntimeError(f"Roster override could not find {player_name} on {team['name']}")

    kawhi = take(clippers, "Kawhi Leonard")
    ingram = take(toronto, "Brandon Ingram")
    dick = take(toronto, "Gradey Dick")
    toronto["players"].append(kawhi)
    clippers["players"].extend((ingram, dick))

    for team in (toronto, clippers):
        team["players"].sort(key=lambda player: (-player["rating"], player["name"]))
        for index, player in enumerate(team["players"], start=1):
            player["modelRank"] = index
        team["rosterCount"] = len(team["players"])
        team["topTen"] = [player["name"] for player in team["players"][:10]]


def read_shots(
    stream: io.TextIOBase,
    team_name_to_abbr: dict[str, str],
) -> tuple[
    dict[str, dict[str, dict[str, int]]],
    dict[str, dict[str, dict[str, int]]],
    dict[str, dict[str, int]],
    int,
]:
    offense: dict[str, dict[str, dict[str, int]]] = defaultdict(empty_zone_counts)
    defense: dict[str, dict[str, dict[str, int]]] = defaultdict(empty_zone_counts)
    league = empty_zone_counts()
    skipped_team_rows = 0

    for row in csv.DictReader(stream):
        zone = SHOT_ZONE_MAP.get(row["SHOT_ZONE_BASIC"])
        if zone is None:
            continue
        made = int(number(row["SHOT_MADE_FLAG"]))
        player_key = normalized_name(row["PLAYER_NAME"])
        add_shot(offense[player_key], zone, made)
        add_shot(league, zone, made)

        offense_abbr = team_name_to_abbr.get(normalized_name(row["TEAM_NAME"]))
        if offense_abbr is None:
            skipped_team_rows += 1
            continue
        home_abbr = NBA_TO_ROSTER_ABBR.get(row["HTM"], row["HTM"])
        away_abbr = NBA_TO_ROSTER_ABBR.get(row["VTM"], row["VTM"])
        defense_abbr = away_abbr if offense_abbr == home_abbr else home_abbr
        add_shot(defense[defense_abbr], zone, made)

    return offense, defense, league, skipped_team_rows


def read_matchups(stream: io.TextIOBase) -> dict[str, dict[str, float]]:
    defenders: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in csv.DictReader(stream):
        # The feed's leading person fields describe the offensive player; the
        # matchups_* fields identify the primary defender for these results.
        key = normalized_name(
            f"{row['matchups_first_name']} {row['matchups_family_name']}"
        )
        target = defenders[key]
        target["possessions"] += number(row["partial_possessions"])
        target["seconds"] += number(row["matchup_minutes_sort"])
        target["points"] += number(row["player_points"])
        target["fgm"] += number(row["matchup_field_goals_made"])
        target["fga"] += number(row["matchup_field_goals_attempted"])
        target["threePm"] += number(row["matchup_three_pointers_made"])
        target["threePa"] += number(row["matchup_three_pointers_attempted"])
        target["turnovers"] += number(row["matchup_turnovers"])
        target["blocks"] += number(row["matchup_blocks"])
    return defenders


def matchup_row(values: dict[str, float] | None) -> dict[str, Any]:
    if not values:
        return {
            "possessions": 0,
            "minutes": 0,
            "points": 0,
            "fgm": 0,
            "fga": 0,
            "fgPct": None,
            "twoPm": 0,
            "twoPa": 0,
            "twoPct": None,
            "threePm": 0,
            "threePa": 0,
            "threePct": None,
            "turnovers": 0,
            "blocks": 0,
        }
    fgm = round(values["fgm"])
    fga = round(values["fga"])
    three_pm = round(values["threePm"])
    three_pa = round(values["threePa"])
    two_pm = fgm - three_pm
    two_pa = fga - three_pa
    return {
        "possessions": round(values["possessions"], 1),
        "minutes": round(values["seconds"] / 60, 1),
        "points": round(values["points"]),
        "fgm": fgm,
        "fga": fga,
        "fgPct": percentage(fgm, fga),
        "twoPm": two_pm,
        "twoPa": two_pa,
        "twoPct": percentage(two_pm, two_pa),
        "threePm": three_pm,
        "threePa": three_pa,
        "threePct": percentage(three_pm, three_pa),
        "turnovers": round(values["turnovers"]),
        "blocks": round(values["blocks"]),
    }


def projected_units(players: list[dict[str, Any]]) -> dict[str, list[str]]:
    available = [player for player in players if player.get("status") == "Active"]
    ordered = sorted(available, key=lambda player: (-player["rating"], player["name"]))
    rotation = ordered[:10]
    return {
        "starters": [player["id"] for player in rotation[:5]],
        "secondUnit": [player["id"] for player in rotation[5:10]],
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.rosters.read_text())
    teams = source["teams"]
    apply_kawhi_trade(teams)
    team_name_to_abbr = {
        normalized_name(team["name"]): team["abbreviation"] for team in teams
    }

    with open_csv(args.shotdetail, SHOT_URL, "shotdetail_2025.csv") as stream:
        offense, team_defense, league, skipped_team_rows = read_shots(
            stream, team_name_to_abbr
        )
    with open_csv(args.matchups, MATCHUP_URL, "matchups_2025.csv") as stream:
        defenders = read_matchups(stream)

    matched_offense = 0
    matched_defense = 0
    output_teams: list[dict[str, Any]] = []
    for team in teams:
        players: list[dict[str, Any]] = []
        for player in team["players"]:
            key = normalized_name(player["name"])
            offense_counts = offense.get(key, empty_zone_counts())
            defense_values = defenders.get(key)
            offense_attempts = sum(item["attempts"] for item in offense_counts.values())
            if offense_attempts:
                matched_offense += 1
            if defense_values and defense_values["fga"]:
                matched_defense += 1
            players.append(
                {
                    "id": player["id"],
                    "name": player["name"],
                    "jersey": player["jersey"],
                    "position": player["position"],
                    "positions": player["positions"],
                    "rating": player["rating"],
                    "headshotUrl": player["headshotUrl"],
                    "headshotVerified": player["headshotVerified"],
                    "status": player["status"],
                    "offense": {
                        "attempts": offense_attempts,
                        "made": sum(item["made"] for item in offense_counts.values()),
                        "zones": zone_rows(offense_counts, league),
                    },
                    "defense": matchup_row(defense_values),
                }
            )

        output_teams.append(
            {
                "id": team["id"],
                "name": team["name"],
                "shortName": team["shortName"],
                "abbreviation": team["abbreviation"],
                "slug": team["slug"],
                "logoUrl": team["logoUrl"],
                "color": team["color"],
                "players": players,
                "projected": projected_units(players),
                "defenseZones": zone_rows(
                    team_defense.get(team["abbreviation"], empty_zone_counts()), league
                ),
            }
        )

    output_teams.sort(key=lambda team: team["name"])
    output = {
        "metadata": {
            "statsSeason": "2025-26",
            "rosterSeason": source["metadata"]["season"],
            "sourceRosterGeneratedAt": source["metadata"]["generatedAt"],
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "rosterAuthority": source["metadata"]["rosterAuthority"],
            "headshotAuthority": source["metadata"]["headshotAuthority"],
            "shotSource": SHOT_URL,
            "matchupSource": MATCHUP_URL,
            "teamCount": len(output_teams),
            "playerCount": sum(len(team["players"]) for team in output_teams),
            "playersWithOffense": matched_offense,
            "playersWithDefense": matched_defense,
            "skippedTeamShotRows": skipped_team_rows,
        },
        "teams": output_teams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps(output["metadata"], indent=2))


if __name__ == "__main__":
    main()
