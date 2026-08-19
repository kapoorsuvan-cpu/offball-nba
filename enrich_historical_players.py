#!/usr/bin/env python3
"""Build precise historical positions and ESPN headshots for dashboard rows.

The model matrix intentionally freezes each team at opening night. This script
uses the same opening-week ESPN box scores to recover stable athlete IDs,
published headshots, and player links. ESPN's generic G/F labels are not shown
as invented PG/SG or SF/PF combinations: specific ESPN positions are preferred,
then the season roster, then the player's HoopsHype profile.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from scrape_dataset import (
    ESPN_OPENING_TEAM_ALIASES,
    OPENING_ROSTER_CUTOFFS,
    TEAMS,
    get_text,
    normalize_name as source_normalize_name,
    scrape_global_ratings,
    scrape_profile_position,
    scrape_wikipedia_positions,
)


START_END_YEAR = 2017
END_END_YEAR = 2026
PRECISE_POSITIONS = {"PG", "SG", "SF", "PF", "C"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
PROFILE_NAME_ALIASES = {
    "Svi Mykhailiuk": "Sviatoslav Mykhailiuk",
    "Herbert Jones": "Herb Jones",
    "Ronald Holland II": "Ron Holland",
    "Bub Carrington": "Carlton Carrington",
}
OFFICIAL_HEADSHOT_FALLBACKS = {
    source_normalize_name("Jalen Hood-Schifino"): {
        "url": "https://cdn.nba.com/headshots/nba/latest/1040x760/1641720.png",
        "source": "NBA official headshot",
    }
}


def normalize_name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).strip()


def loose_name(value: str) -> str:
    words = [word for word in normalize_name(value).split() if word not in SUFFIXES]
    return " ".join(words)


def precise_position(value: Any) -> str | None:
    position = str(value or "").upper().strip()
    return position if position in PRECISE_POSITIONS else None


def athlete_link(athlete: dict[str, Any]) -> str | None:
    return next(
        (
            str(link["href"])
            for link in athlete.get("links", [])
            if "athlete" in link.get("rel", []) and link.get("href")
        ),
        None,
    )


def opening_week_metadata(
    years: range,
) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    exact: dict[tuple[int, str, str], dict[str, Any]] = {}
    global_players: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_global: set[tuple[str, str]] = set()
    for end_year in years:
        first_day = datetime.fromisoformat(
            OPENING_ROSTER_CUTOFFS[end_year].replace("Z", "+00:00")
        )
        events = json.loads(
            get_text(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
                params={
                    "dates": f"{first_day:%Y%m%d}-{(first_day + timedelta(days=7)):%Y%m%d}",
                    "limit": 1000,
                },
            )
        ).get("events", [])
        for event_id in sorted({str(event["id"]) for event in events}):
            summary = json.loads(
                get_text(
                    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                    params={"event": event_id},
                )
            )
            for block in summary.get("boxscore", {}).get("players", []):
                source_team = str(block["team"]["displayName"])
                team = ESPN_OPENING_TEAM_ALIASES.get(source_team, source_team)
                statistics = block.get("statistics") or []
                if not statistics:
                    continue
                for item in statistics[0].get("athletes", []):
                    athlete = item.get("athlete") or {}
                    name = str(athlete.get("displayName") or "").strip()
                    athlete_id = str(athlete.get("id") or "").strip()
                    if not name or not athlete_id:
                        continue
                    published_headshot = (athlete.get("headshot") or {}).get("href")
                    record = {
                        "name": name,
                        "espnId": athlete_id,
                        "espnUrl": athlete_link(athlete)
                        or f"https://www.espn.com/nba/player/_/id/{athlete_id}",
                        "headshotUrl": published_headshot
                        or f"https://a.espncdn.com/i/headshots/nba/players/full/{athlete_id}.png",
                        "headshotPublished": bool(published_headshot),
                        "espnPosition": (athlete.get("position") or {}).get("abbreviation"),
                    }
                    source_key = source_normalize_name(name)
                    exact.setdefault((end_year, team, source_key), record)
                    identity = (source_key, athlete_id)
                    if identity not in seen_global:
                        global_players[source_key].append(record)
                        seen_global.add(identity)
    return exact, global_players


def unique_global_match(
    name: str,
    players: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    source_key = source_normalize_name(name)
    candidates = players.get(source_key, [])
    if not candidates:
        loose = loose_name(name)
        candidates = [
            record
            for key, records in players.items()
            if loose_name(key) == loose
            for record in records
        ]
    ids = {record["espnId"] for record in candidates}
    if len(ids) != 1:
        return None
    return next(
        (record for record in reversed(candidates) if record["headshotPublished"]),
        candidates[-1],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data/nba_2k_team_seasons_backtest.csv")
    parser.add_argument("--output", default="data/historical_player_metadata.json")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    history = pd.read_csv(args.history).query(
        "@START_END_YEAR <= season_end_year <= @END_END_YEAR"
    )
    years = range(START_END_YEAR, END_END_YEAR + 1)
    espn_exact, espn_global = opening_week_metadata(years)

    wiki_positions: dict[tuple[int, str], dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(scrape_wikipedia_positions, team, end_year): (end_year, team.name)
            for end_year in years
            for team in TEAMS
        }
        for future in as_completed(futures):
            wiki_positions[futures[future]] = future.result()

    profile_urls: dict[str, str] = {}
    for end_year in years:
        for player in scrape_global_ratings(end_year):
            profile_urls.setdefault(
                source_normalize_name(player["name"]), player["profile_url"]
            )
    for history_name, profile_name in PROFILE_NAME_ALIASES.items():
        profile_url = profile_urls.get(source_normalize_name(profile_name))
        if profile_url:
            profile_urls[source_normalize_name(history_name)] = profile_url

    position_history: dict[str, Counter[str]] = defaultdict(Counter)
    for record in espn_exact.values():
        position = precise_position(record.get("espnPosition"))
        if position:
            position_history[source_normalize_name(record["name"])][position] += 1
    for positions in wiki_positions.values():
        for player_name, raw_position in positions.items():
            position = precise_position(raw_position)
            if position:
                position_history[source_normalize_name(player_name)][position] += 1

    unresolved_names: set[str] = set()
    history_rows: list[tuple[int, str, str]] = []
    for _, row in history.iterrows():
        end_year = int(row["season_end_year"])
        team = str(row["team"])
        for rank in range(1, 11):
            name = str(row[f"player_{rank}"])
            strict = source_normalize_name(name)
            history_rows.append((end_year, team, name))
            espn = espn_exact.get((end_year, team, strict))
            wiki = wiki_positions.get((end_year, team), {}).get(strict)
            if not precise_position((espn or {}).get("espnPosition")) and not precise_position(wiki):
                unresolved_names.add(strict)

    profile_positions: dict[str, str] = {}
    profile_jobs = {
        name: profile_urls[name]
        for name in unresolved_names
        if name in profile_urls and profile_urls[name]
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(scrape_profile_position, url): name
            for name, url in profile_jobs.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                profile_positions[name] = future.result()
            except Exception:
                profile_positions[name] = "UNK"

    output_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    missing_headshots: list[str] = []
    unresolved_positions: list[str] = []
    for end_year, team, name in history_rows:
        strict = source_normalize_name(name)
        exact_record = espn_exact.get((end_year, team, strict))
        global_record = unique_global_match(name, espn_global)
        athlete = exact_record or global_record
        if athlete is None:
            missing_headshots.append(f"{end_year}|{team}|{name}")
        official_headshot = OFFICIAL_HEADSHOT_FALLBACKS.get(strict)
        headshot_url = (
            official_headshot["url"]
            if official_headshot
            else athlete.get("headshotUrl")
            if athlete
            else None
        )
        headshot_verified = bool(
            official_headshot or (athlete and athlete.get("headshotPublished"))
        )
        headshot_source = (
            official_headshot["source"]
            if official_headshot
            else "ESPN opening box score"
            if exact_record
            else "ESPN cross-season athlete ID"
            if athlete
            else None
        )

        wiki_position = wiki_positions.get((end_year, team), {}).get(strict)
        position = precise_position((exact_record or {}).get("espnPosition"))
        if position:
            position_source = "ESPN opening box score"
        else:
            position = precise_position(wiki_position)
            if position:
                position_source = "Wikipedia season roster"
            else:
                position = precise_position(profile_positions.get(strict))
                if position:
                    position_source = "HoopsHype player profile"
                else:
                    common = position_history.get(strict)
                    position = common.most_common(1)[0][0] if common else None
                    position_source = "Cross-season position fallback"
        if not position:
            unresolved_positions.append(f"{end_year}|{team}|{name}")
            position = "SF"
            position_source = "Unresolved fallback"
        source_counts[position_source] += 1

        output_rows.append(
            {
                "seasonEndYear": end_year,
                "team": team,
                "player": name,
                "normalizedName": normalize_name(name),
                "position": position,
                "positionSource": position_source,
                "espnId": athlete.get("espnId") if athlete else None,
                "espnUrl": athlete.get("espnUrl") if athlete else None,
                "headshotUrl": headshot_url,
                "headshotPublished": bool(athlete and athlete.get("headshotPublished")),
                "headshotVerified": headshot_verified,
                "headshotSource": headshot_source,
            }
        )

    if missing_headshots:
        raise RuntimeError(f"Missing historical ESPN matches: {missing_headshots[:20]}")
    if unresolved_positions:
        raise RuntimeError(f"Unresolved historical positions: {unresolved_positions[:20]}")
    if len(output_rows) != 3000:
        raise RuntimeError(f"Expected 3,000 historical player rows, got {len(output_rows)}")

    payload = {
        "metadata": {
            "seasons": [f"{year - 1}-{str(year)[-2:]}" for year in years],
            "playerRows": len(output_rows),
            "uniquePlayers": len({row["espnId"] for row in output_rows}),
            "headshotRows": sum(bool(row["headshotUrl"]) for row in output_rows),
            "publishedHeadshotRows": sum(row["headshotPublished"] for row in output_rows),
            "verifiedHeadshotRows": sum(row["headshotVerified"] for row in output_rows),
            "officialFallbackHeadshotRows": sum(
                row["headshotSource"] == "NBA official headshot" for row in output_rows
            ),
            "positionSourceCounts": dict(source_counts),
        },
        "players": output_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
