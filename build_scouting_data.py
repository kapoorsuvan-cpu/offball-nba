#!/usr/bin/env python3
"""Build the compact data bundle used by the OFFBALL scouting dashboard.

The roster snapshot is sourced from the existing OFFBALL data file. Shooting
and matchup results are sourced from shufinskiy/nba_data for the 2025-26
regular season. Explicit roster, availability, and coaching overrides keep the
published rotations aligned with the requested 2026-27 scouting assumptions.
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
from html.parser import HTMLParser
from urllib.error import URLError
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_ROSTERS = ROOT / "dashboard-app/app/data/current-rosters.json"
DEFAULT_RATINGS = ROOT / "data/2k27_current_ratings.browser.json"
DEFAULT_OUTPUT = ROOT / "dashboard-app/app/scout/data/scouting-data.json"
ROSTER_SNAPSHOT_AT = "2026-08-20T00:00:00+00:00"

SHOT_URL = (
    "https://raw.githubusercontent.com/shufinskiy/nba_data/main/"
    "datasets/shotdetail_2025.tar.xz"
)
MATCHUP_URL = (
    "https://raw.githubusercontent.com/shufinskiy/nba_data/main/"
    "datasets/matchups_2025.tar.xz"
)
ESPN_PLAYER_STATS_URL = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/"
    "statistics/byathlete?region=us&lang=en&contentorigin=espn&isqualified=false&"
    "page=1&limit=1000&sort=offensive.avgPoints:desc&season=2026&seasontype=2"
)
ADVANCED_STATS_URL = "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html"
DEPTH_CHART_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vTi9up0zyRwtsmYQjpMgyUVvR0LMhiG76bZkhe4V7dw7pxf6wm2jww_"
    "fxzCijIXFN-ogn-CqUhjj2l0/pub?gid=699250664&single=true&output=csv"
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

DEPTH_TEAM_ABBREVIATIONS = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GS",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NO",
    "NEW YORK KNICKS": "NY",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAILBLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SA",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTAH",
    "WASHINGTON WIZARDS": "WSH",
}

DEPTH_NAME_ALIASES = {
    "egordmin": "Egor Demin",
    "dennisschroder": "Dennis Schroder",
    "morezjohnson": "Morez Johnson Jr.",
    "ronholland": "Ronald Holland II",
    "jimmybutler": "Jimmy Butler III",
    "robertwilliams": "Robert Williams III",
    "dariusacuff": "Darius Acuff Jr.",
    "ggjacksonii": "GG Jackson",
}

# The published depth chart is the baseline. These explicit user decisions
# replace one named slot without re-ranking the rest of the unit.
DEPTH_UNIT_REPLACEMENTS = {
    "CLE": {"starters": {"Sam Merrill": "James Harden"}},
    "LAL": {"secondUnit": {"Cameron Carr": "Matisse Thybulle"}},
    "MIA": {"secondUnit": {"Nikola Jovic": "Nick Richards"}},
    "PHX": {"secondUnit": {"Luke Kennard": "Haywood Highsmith"}},
}

DEPTH_ROW_POSITIONS = (
    frozenset(("C",)),
    frozenset(("PF", "C")),
    frozenset(("SF", "PF")),
    frozenset(("SG", "SF")),
    frozenset(("PG", "SG")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosters", type=Path, default=DEFAULT_ROSTERS)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--depth-charts", type=Path)
    parser.add_argument("--shotdetail", type=Path)
    parser.add_argument("--matchups", type=Path)
    parser.add_argument("--player-stats", type=Path)
    parser.add_argument("--advanced-stats", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def normalized_name(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_value.lower())


def stat_name_key(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    without_suffix = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", ascii_value.lower())
    return re.sub(r"[^a-z0-9]", "", without_suffix)


def number(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def download_text(url: str, timeout: int = 90) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "offball-scout/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except URLError:
        result = subprocess.run(
            ["curl", "-fsSL", "-A", "offball-scout/1.0", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout


def open_json(path: Path | None, url: str) -> dict[str, Any]:
    if path:
        return json.loads(path.read_text())
    return json.loads(download_text(url))


def open_text(path: Path | None, url: str) -> str:
    if path:
        return path.read_text()
    return download_text(url)


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


def open_depth_chart_csv(path: Path | None) -> io.TextIOBase:
    if path:
        return path.open(encoding="utf-8-sig", newline="")
    request = urllib.request.Request(
        DEPTH_CHART_URL, headers={"User-Agent": "offball-scout/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            csv_text = response.read().decode("utf-8-sig")
    except URLError:
        result = subprocess.run(
            ["curl", "-fsSL", DEPTH_CHART_URL],
            check=True,
            capture_output=True,
            text=True,
        )
        csv_text = result.stdout
    return io.StringIO(csv_text)


def clean_depth_name(value: str) -> str | None:
    name = re.sub(r"\s*\(\d+\)\s*$", "", value).strip()
    if not name or name.upper().startswith("OPEN"):
        return None
    return DEPTH_NAME_ALIASES.get(normalized_name(name), name)


def read_depth_charts(stream: io.TextIOBase) -> dict[str, list[dict[str, str | None]]]:
    charts: dict[str, list[dict[str, str | None]]] = {}
    active_abbreviation: str | None = None
    remaining_rows = 0
    for row in csv.reader(stream):
        if not row:
            continue
        heading = row[0].strip()
        if heading in DEPTH_TEAM_ABBREVIATIONS:
            active_abbreviation = DEPTH_TEAM_ABBREVIATIONS[heading]
            charts[active_abbreviation] = []
            remaining_rows = 0
            continue
        if active_abbreviation and heading in {"STARTERS", "Player"}:
            remaining_rows = 5
            continue
        if not active_abbreviation or remaining_rows == 0:
            continue
        padded = row + [""] * (13 - len(row))
        charts[active_abbreviation].append(
            {
                "starter": clean_depth_name(padded[0]),
                "second": clean_depth_name(padded[4]),
                "third": clean_depth_name(padded[8]),
                "other": clean_depth_name(padded[12]),
            }
        )
        remaining_rows -= 1

    if len(charts) != 30:
        raise RuntimeError(f"Expected 30 NBA depth charts, received {len(charts)}")
    malformed = [abbr for abbr, rows in charts.items() if len(rows) != 5]
    if malformed:
        raise RuntimeError(f"Depth charts did not contain five position rows: {malformed}")
    return charts


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


def apply_payton_watson_trade(teams: list[dict[str, Any]]) -> None:
    """Apply the confirmed five-team player movement supplied by the user."""
    by_abbreviation = {team["abbreviation"]: team for team in teams}

    def take(player_name: str) -> dict[str, Any]:
        for team in teams:
            for index, player in enumerate(team["players"]):
                if player["name"] == player_name:
                    return team["players"].pop(index)
        raise RuntimeError(f"Trade override could not find {player_name}")

    def send(player_name: str, destination: str) -> None:
        by_abbreviation[destination]["players"].append(take(player_name))

    send("Peyton Watson", "CLE")
    send("Cam Whitmore", "LAC")
    send("Max Strus", "LAC")
    send("Tre Mann", "WSH")
    send("Dennis Schroder", "CHA")

    # Denver receives Julian Reese in the trade, then waives him as reported.
    take("Julian Reese")


def current_rating_index(rating_teams: list[dict[str, Any]]) -> dict[str, int]:
    ratings: dict[str, int] = {}
    for team in rating_teams:
        for player in team["players"]:
            ratings[normalized_name(player["player"])] = int(player["rating"])
    return ratings


def refresh_player_ratings(
    teams: list[dict[str, Any]], rating_teams: list[dict[str, Any]]
) -> None:
    ratings = current_rating_index(rating_teams)
    for team in teams:
        for player in team["players"]:
            rating = ratings.get(normalized_name(player["name"]))
            if rating is not None:
                player["rating"] = rating


def apply_scouting_overrides(teams: list[dict[str, Any]]) -> None:
    by_abbreviation = {team["abbreviation"]: team for team in teams}
    miami = by_abbreviation["MIA"]
    if not any(player["name"] == "Nick Richards" for player in miami["players"]):
        miami["players"].append(
            {
                "id": "4278076",
                "name": "Nick Richards",
                "jersey": None,
                "position": "C",
                "positions": ["C"],
                "rating": 75,
                "headshotUrl": (
                    "https://a.espncdn.com/i/headshots/nba/players/full/4278076.png"
                ),
                "headshotVerified": True,
                "status": "Active",
            }
        )

    phoenix = by_abbreviation["PHX"]
    if not any(player["name"] == "Haywood Highsmith" for player in phoenix["players"]):
        phoenix["players"].append(
            {
                "id": "4291678",
                "name": "Haywood Highsmith",
                "jersey": "7",
                "position": "SF",
                "positions": ["SF", "PF"],
                "rating": 73,
                "headshotUrl": (
                    "https://a.espncdn.com/i/headshots/nba/players/full/4291678.png"
                ),
                "headshotVerified": True,
                "status": "Active",
            }
        )

    for player in by_abbreviation["GS"]["players"]:
        if player["name"] == "Jimmy Butler III":
            player["status"] = "Out · right ACL rehabilitation"
            break
    else:
        raise RuntimeError("Availability override could not find Jimmy Butler III")

    # Keep the user-specified value independent of later source refreshes.
    richards = next(
        player for player in miami["players"] if player["name"] == "Nick Richards"
    )
    richards["rating"] = 75
    highsmith = next(
        player for player in phoenix["players"] if player["name"] == "Haywood Highsmith"
    )
    highsmith["rating"] = 73
    watson = next(
        player
        for team in teams
        for player in team["players"]
        if player["name"] == "Peyton Watson"
    )
    watson["rating"] = 81


def reconcile_depth_chart_rosters(
    teams: list[dict[str, Any]],
    depth_charts: dict[str, list[dict[str, str | None]]],
) -> None:
    by_abbreviation = {team["abbreviation"]: team for team in teams}
    owners: dict[str, dict[str, Any]] = {}
    players_by_key: dict[str, dict[str, Any]] = {}
    for team in teams:
        for player in team["players"]:
            key = normalized_name(player["name"])
            owners[key] = team
            players_by_key[key] = player

    for abbreviation, rows in depth_charts.items():
        target = by_abbreviation[abbreviation]
        for row in rows:
            for tier in ("starter", "second", "third", "other"):
                name = row[tier]
                if not name:
                    continue
                key = normalized_name(name)
                player = players_by_key.get(key)
                owner = owners.get(key)
                if player is None or owner is None or owner is target:
                    continue
                owner["players"].remove(player)
                target["players"].append(player)
                owners[key] = target


def rerank_rosters(teams: list[dict[str, Any]]) -> None:
    for team in teams:
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
    dict[str, str],
    int,
]:
    offense: dict[str, dict[str, dict[str, int]]] = defaultdict(empty_zone_counts)
    defense: dict[str, dict[str, dict[str, int]]] = defaultdict(empty_zone_counts)
    league = empty_zone_counts()
    nba_player_ids: dict[str, str] = {}
    skipped_team_rows = 0

    for row in csv.DictReader(stream):
        zone = SHOT_ZONE_MAP.get(row["SHOT_ZONE_BASIC"])
        if zone is None:
            continue
        made = int(number(row["SHOT_MADE_FLAG"]))
        player_key = normalized_name(row["PLAYER_NAME"])
        nba_player_ids[player_key] = row["PLAYER_ID"]
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

    return offense, defense, league, nba_player_ids, skipped_team_rows


def read_player_stats(
    payload: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    category_names = {
        category["name"]: category["names"] for category in payload["categories"]
    }
    by_id: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for item in payload["athletes"]:
        values: dict[str, Any] = {}
        for category in item["categories"]:
            names = category_names.get(category["name"], [])
            values.update(dict(zip(names, category["values"])))
        athlete = item["athlete"]
        stats = {"values": values, "headshotUrl": athlete.get("headshot")}
        by_id[str(athlete["id"])] = stats
        by_name[stat_name_key(athlete["displayName"])] = stats
    return by_id, by_name


class AdvancedStatsParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.current_row: dict[str, str] | None = None
        self.current_stat: str | None = None
        self.cell_parts: list[str] = []
        self.rows: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "advanced":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.current_row = {}
        elif self.in_table and tag in {"td", "th"} and self.current_row is not None:
            self.current_stat = attributes.get("data-stat")
            self.cell_parts = []

    def handle_data(self, data: str) -> None:
        if self.current_stat is not None:
            self.cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current_row is not None and self.current_stat:
            self.current_row[self.current_stat] = "".join(self.cell_parts).strip()
            self.current_stat = None
            self.cell_parts = []
        elif tag == "tr" and self.current_row is not None:
            if self.current_row.get("name_display") not in {None, "Player", "League Average"}:
                self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.in_table:
            self.in_table = False


def read_advanced_stats(html: str) -> dict[str, dict[str, float | None]]:
    parser = AdvancedStatsParser()
    parser.feed(html)
    advanced: dict[str, dict[str, float | None]] = {}
    games_by_name: dict[str, int] = {}
    for row in parser.rows:
        key = stat_name_key(row["name_display"])
        games = int(optional_number(row.get("games")) or 0)
        # If Basketball Reference includes team splits, retain the aggregate or
        # the row with the largest sample.
        if key in advanced and games <= games_by_name[key]:
            continue
        ts_pct = optional_number(row.get("ts_pct"))
        advanced[key] = {
            "tsPct": round(ts_pct * 100, 1) if ts_pct is not None else None,
            "per": optional_number(row.get("per")),
            "usagePct": optional_number(row.get("usg_pct")),
            "bpm": optional_number(row.get("bpm")),
        }
        games_by_name[key] = games
    return advanced


def season_stats_row(
    traditional: dict[str, Any] | None,
    advanced: dict[str, float | None] | None,
) -> dict[str, Any]:
    empty = {
        "games": 0,
        "minutes": None,
        "points": None,
        "rebounds": None,
        "assists": None,
        "steals": None,
        "blocks": None,
        "fgPct": None,
        "threePct": None,
        "ftPct": None,
        "tsPct": None,
        "effectiveFgPct": None,
        "assistTurnoverRatio": None,
        "usagePct": None,
        "per": None,
        "bpm": None,
    }
    if traditional is None:
        return {**empty, **(advanced or {})}

    values = traditional["values"]
    fgm = optional_number(values.get("fieldGoalsMade"))
    fga = optional_number(values.get("fieldGoalsAttempted"))
    three_pm = optional_number(values.get("threePointFieldGoalsMade"))
    points = optional_number(values.get("points"))
    fta = optional_number(values.get("freeThrowsAttempted"))
    assists = optional_number(values.get("assists"))
    turnovers = optional_number(values.get("turnovers"))
    ts_denominator = 2 * (fga + 0.44 * fta) if fga is not None and fta is not None else 0

    return {
        "games": int(optional_number(values.get("gamesPlayed")) or 0),
        "minutes": round(optional_number(values.get("avgMinutes")) or 0, 1),
        "points": round(optional_number(values.get("avgPoints")) or 0, 1),
        "rebounds": round(optional_number(values.get("avgRebounds")) or 0, 1),
        "assists": round(optional_number(values.get("avgAssists")) or 0, 1),
        "steals": round(optional_number(values.get("avgSteals")) or 0, 1),
        "blocks": round(optional_number(values.get("avgBlocks")) or 0, 1),
        "fgPct": round(optional_number(values.get("fieldGoalPct")) or 0, 1),
        "threePct": round(optional_number(values.get("threePointFieldGoalPct")) or 0, 1),
        "ftPct": round(optional_number(values.get("freeThrowPct")) or 0, 1),
        "tsPct": round(100 * points / ts_denominator, 1)
        if points is not None and ts_denominator
        else None,
        "effectiveFgPct": round(100 * (fgm + 0.5 * three_pm) / fga, 1)
        if fgm is not None and three_pm is not None and fga
        else None,
        "assistTurnoverRatio": round(assists / turnovers, 2)
        if assists is not None and turnovers
        else None,
        "usagePct": None,
        "per": None,
        "bpm": None,
        **(advanced or {}),
    }


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


def best_depth_fallback(
    players: list[dict[str, Any]],
    accepted_positions: frozenset[str],
    selected_ids: set[str],
) -> dict[str, Any]:
    available = [
        player
        for player in players
        if player.get("status") == "Active" and player["id"] not in selected_ids
    ]
    if not available:
        raise RuntimeError("No available player remained to complete a depth-chart unit")

    def score(player: dict[str, Any]) -> tuple[int, int, str]:
        positions = player["positions"]
        primary_fit = bool(positions and positions[0] in accepted_positions)
        any_fit = bool(accepted_positions.intersection(positions))
        fit_score = 2 if primary_fit else 1 if any_fit else 0
        return (fit_score, int(player["rating"]), player["name"])

    return max(available, key=score)


def replace_depth_slot(
    unit: list[dict[str, Any]],
    old_name: str,
    new_name: str,
    players_by_name: dict[str, dict[str, Any]],
) -> None:
    try:
        index = next(i for i, player in enumerate(unit) if player["name"] == old_name)
    except StopIteration as error:
        raise RuntimeError(f"Depth-chart override could not find {old_name}") from error
    incoming = players_by_name.get(normalized_name(new_name))
    if incoming is None:
        raise RuntimeError(f"Depth-chart override could not find {new_name}")
    if incoming.get("status") != "Active":
        raise RuntimeError(f"Depth-chart override requires unavailable player {new_name}")
    unit[index] = incoming


def projected_units(
    team_abbreviation: str,
    players: list[dict[str, Any]],
    depth_rows: list[dict[str, str | None]],
) -> dict[str, list[str]]:
    players_by_name = {normalized_name(player["name"]): player for player in players}
    selected_ids: set[str] = set()
    starters_by_row: list[dict[str, Any]] = []

    def choose(names: Iterable[str | None]) -> dict[str, Any] | None:
        for name in names:
            if not name:
                continue
            player = players_by_name.get(normalized_name(name))
            if (
                player
                and player.get("status") == "Active"
                and player["id"] not in selected_ids
            ):
                return player
        return None

    for row_index, row in enumerate(depth_rows):
        player = choose((row["starter"], row["second"], row["third"], row["other"]))
        if player is None:
            player = best_depth_fallback(
                players, DEPTH_ROW_POSITIONS[row_index], selected_ids
            )
        starters_by_row.append(player)
        selected_ids.add(player["id"])

    second_unit_by_row: list[dict[str, Any]] = []
    for row_index, row in enumerate(depth_rows):
        player = choose((row["second"], row["third"], row["other"], row["starter"]))
        if player is None:
            player = best_depth_fallback(
                players, DEPTH_ROW_POSITIONS[row_index], selected_ids
            )
        second_unit_by_row.append(player)
        selected_ids.add(player["id"])

    # The source sheet is ordered C-to-PG; the UI reads naturally PG-to-C.
    starters = list(reversed(starters_by_row))
    second_unit = list(reversed(second_unit_by_row))
    replacements = DEPTH_UNIT_REPLACEMENTS.get(team_abbreviation, {})
    for old_name, new_name in replacements.get("starters", {}).items():
        replace_depth_slot(starters, old_name, new_name, players_by_name)
    for old_name, new_name in replacements.get("secondUnit", {}).items():
        replace_depth_slot(second_unit, old_name, new_name, players_by_name)

    starter_ids = {player["id"] for player in starters}
    second_ids = {player["id"] for player in second_unit}
    if len(starter_ids) != 5 or len(second_ids) != 5 or starter_ids & second_ids:
        raise RuntimeError(f"{team_abbreviation} depth chart did not produce ten unique players")
    return {
        "starters": [player["id"] for player in starters],
        "secondUnit": [player["id"] for player in second_unit],
    }


def main() -> None:
    args = parse_args()
    source = json.loads(args.rosters.read_text())
    rating_teams = json.loads(args.ratings.read_text())
    with open_depth_chart_csv(args.depth_charts) as stream:
        depth_charts = read_depth_charts(stream)
    teams = source["teams"]
    apply_kawhi_trade(teams)
    refresh_player_ratings(teams, rating_teams)
    apply_scouting_overrides(teams)
    reconcile_depth_chart_rosters(teams, depth_charts)
    apply_payton_watson_trade(teams)
    rerank_rosters(teams)
    team_name_to_abbr = {
        normalized_name(team["name"]): team["abbreviation"] for team in teams
    }

    with open_csv(args.shotdetail, SHOT_URL, "shotdetail_2025.csv") as stream:
        offense, team_defense, league, nba_player_ids, skipped_team_rows = read_shots(
            stream, team_name_to_abbr
        )
    with open_csv(args.matchups, MATCHUP_URL, "matchups_2025.csv") as stream:
        defenders = read_matchups(stream)
    player_stats_by_id, player_stats_by_name = read_player_stats(
        open_json(args.player_stats, ESPN_PLAYER_STATS_URL)
    )
    advanced_stats = read_advanced_stats(
        open_text(args.advanced_stats, ADVANCED_STATS_URL)
    )

    matched_offense = 0
    matched_defense = 0
    matched_season_stats = 0
    output_teams: list[dict[str, Any]] = []
    for team in teams:
        players: list[dict[str, Any]] = []
        for player in team["players"]:
            key = normalized_name(player["name"])
            offense_counts = offense.get(key, empty_zone_counts())
            defense_values = defenders.get(key)
            traditional_stats = player_stats_by_id.get(str(player["id"])) or player_stats_by_name.get(
                stat_name_key(player["name"])
            )
            season_stats = season_stats_row(
                traditional_stats, advanced_stats.get(stat_name_key(player["name"]))
            )
            offense_attempts = sum(item["attempts"] for item in offense_counts.values())
            if offense_attempts:
                matched_offense += 1
            if defense_values and defense_values["fga"]:
                matched_defense += 1
            if season_stats["games"]:
                matched_season_stats += 1
            nba_player_id = nba_player_ids.get(key)
            players.append(
                {
                    "id": player["id"],
                    "name": player["name"],
                    "jersey": player["jersey"],
                    "position": player["position"],
                    "positions": player["positions"],
                    "rating": player["rating"],
                    "headshotUrl": player["headshotUrl"] or (
                        traditional_stats.get("headshotUrl") if traditional_stats else None
                    ),
                    "headshotFallbackUrl": (
                        f"https://cdn.nba.com/headshots/nba/latest/1040x760/{nba_player_id}.png"
                        if nba_player_id
                        else None
                    ),
                    "headshotVerified": player["headshotVerified"],
                    "status": player["status"],
                    "seasonStats": season_stats,
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
                "projected": projected_units(
                    team["abbreviation"], players, depth_charts[team["abbreviation"]]
                ),
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
            "rosterSnapshotAt": ROSTER_SNAPSHOT_AT,
            "sourceRosterGeneratedAt": source["metadata"]["generatedAt"],
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "rosterAuthority": source["metadata"]["rosterAuthority"],
            "headshotAuthority": source["metadata"]["headshotAuthority"],
            "ratingsSource": "https://www.2kratings.com/teams",
            "depthChartSource": "https://www.nbadepthcharts.com",
            "rotationMethod": "NBA Depth Charts starters and second string with availability overrides",
            "shotSource": SHOT_URL,
            "matchupSource": MATCHUP_URL,
            "playerStatsSource": ESPN_PLAYER_STATS_URL,
            "advancedStatsSource": ADVANCED_STATS_URL,
            "teamCount": len(output_teams),
            "playerCount": sum(len(team["players"]) for team in output_teams),
            "playersWithOffense": matched_offense,
            "playersWithDefense": matched_defense,
            "playersWithSeasonStats": matched_season_stats,
            "skippedTeamShotRows": skipped_team_rows,
        },
        "teams": output_teams,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps(output["metadata"], indent=2))


if __name__ == "__main__":
    main()
