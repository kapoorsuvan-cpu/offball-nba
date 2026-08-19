#!/usr/bin/env python3
"""Build an opening-roster matrix of HoopsHype NBA 2K ratings and NBA results.

Player ratings/names come from HoopsHype. Team membership is frozen to each
team's first ESPN regular-season box-score roster (including DNPs), supplemented
only by same-team returnees from the prior opening roster so injured players are
not dropped. In-season trades therefore cannot leak into a preseason forecast.
Season-specific positions prefer that ESPN opening roster, then the corresponding
Wikipedia team-season roster, with the HoopsHype player profile used only as a
documented fallback. Team records come from ESPN's public standings endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


DEFAULT_START_END_YEAR = 2017
DEFAULT_END_END_YEAR = 2026

# UTC cutoffs immediately before the first regular-season game of each season.
# Historical roster template revisions are selected at or before these times.
OPENING_ROSTER_CUTOFFS = {
    2014: "2013-10-29T00:00:00Z",
    2015: "2014-10-28T00:00:00Z",
    2016: "2015-10-27T00:00:00Z",
    2017: "2016-10-25T00:00:00Z",
    2018: "2017-10-17T00:00:00Z",
    2019: "2018-10-16T00:00:00Z",
    2020: "2019-10-22T00:00:00Z",
    2021: "2020-12-22T00:00:00Z",
    2022: "2021-10-19T00:00:00Z",
    2023: "2022-10-18T00:00:00Z",
    2024: "2023-10-24T00:00:00Z",
    2025: "2024-10-22T00:00:00Z",
    2026: "2025-10-21T00:00:00Z",
}


@dataclass(frozen=True)
class Team:
    name: str
    hoopshype_slug: str
    wiki_name: str


TEAMS = (
    Team("Atlanta Hawks", "atlanta-hawks", "Atlanta_Hawks"),
    Team("Boston Celtics", "boston-celtics", "Boston_Celtics"),
    Team("Brooklyn Nets", "brooklyn-nets", "Brooklyn_Nets"),
    Team("Charlotte Hornets", "charlotte-hornets", "Charlotte_Hornets"),
    Team("Chicago Bulls", "chicago-bulls", "Chicago_Bulls"),
    Team("Cleveland Cavaliers", "cleveland-cavaliers", "Cleveland_Cavaliers"),
    Team("Dallas Mavericks", "dallas-mavericks", "Dallas_Mavericks"),
    Team("Denver Nuggets", "denver-nuggets", "Denver_Nuggets"),
    Team("Detroit Pistons", "detroit-pistons", "Detroit_Pistons"),
    Team("Golden State Warriors", "golden-state-warriors", "Golden_State_Warriors"),
    Team("Houston Rockets", "houston-rockets", "Houston_Rockets"),
    Team("Indiana Pacers", "indiana-pacers", "Indiana_Pacers"),
    Team("Los Angeles Clippers", "los-angeles-clippers", "Los_Angeles_Clippers"),
    Team("Los Angeles Lakers", "los-angeles-lakers", "Los_Angeles_Lakers"),
    Team("Memphis Grizzlies", "memphis-grizzlies", "Memphis_Grizzlies"),
    Team("Miami Heat", "miami-heat", "Miami_Heat"),
    Team("Milwaukee Bucks", "milwaukee-bucks", "Milwaukee_Bucks"),
    Team("Minnesota Timberwolves", "minnesota-timberwolves", "Minnesota_Timberwolves"),
    Team("New Orleans Pelicans", "new-orleans-pelicans", "New_Orleans_Pelicans"),
    Team("New York Knicks", "new-york-knicks", "New_York_Knicks"),
    Team("Oklahoma City Thunder", "oklahoma-city-thunder", "Oklahoma_City_Thunder"),
    Team("Orlando Magic", "orlando-magic", "Orlando_Magic"),
    Team("Philadelphia 76ers", "philadelphia-76ers", "Philadelphia_76ers"),
    Team("Phoenix Suns", "phoenix-suns", "Phoenix_Suns"),
    Team("Portland Trail Blazers", "portland-trail-blazers", "Portland_Trail_Blazers"),
    Team("Sacramento Kings", "sacramento-kings", "Sacramento_Kings"),
    Team("San Antonio Spurs", "san-antonio-spurs", "San_Antonio_Spurs"),
    Team("Toronto Raptors", "toronto-raptors", "Toronto_Raptors"),
    Team("Utah Jazz", "utah-jazz", "Utah_Jazz"),
    Team("Washington Wizards", "washington-wizards", "Washington_Wizards"),
)

ESPN_TEAM_NAMES = {"Los Angeles Clippers": "LA Clippers"}
ESPN_OPENING_TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
}


def source_team_name(team: Team, end_year: int) -> str:
    if team.name == "Charlotte Hornets" and end_year <= 2014:
        return "Charlotte Bobcats"
    return ESPN_TEAM_NAMES.get(team.name, team.name)


def source_team_slug(team: Team, end_year: int) -> str:
    if team.name == "Charlotte Hornets" and end_year <= 2014:
        return "charlotte-bobcats"
    return team.hoopshype_slug


def source_wiki_name(team: Team, end_year: int) -> str:
    if team.name == "Charlotte Hornets" and end_year <= 2014:
        return "Charlotte_Bobcats"
    return team.wiki_name


def season_label(end_year: int) -> str:
    return f"{end_year - 1}-{str(end_year)[-2:]}"


def normalize_name(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)\s*", " ", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = value.lower().replace("’", "'").replace(".", "")
    value = re.sub(r"[^a-z0-9]+", "", value)
    suffixes = ("iii", "ii", "iv", "jr", "sr")
    for suffix in suffixes:
        if value.endswith(suffix) and len(value) > len(suffix) + 3:
            value = value[: -len(suffix)]
            break
    return value


def get_text(url: str, *, params: dict[str, Any] | None = None, retries: int = 4) -> str:
    prepared_url = requests.Request("GET", url, params=params).prepare().url or url
    cache_dir = Path(".cache/http")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hashlib.sha256(prepared_url.encode()).hexdigest()}.html"
    if cache_path.exists():
        return cache_path.read_text()
    last_error: Exception | None = None
    headers = None
    if "wikipedia.org" in url:
        headers = {"User-Agent": "nba-2k-win-model/1.0 (educational; local dataset build)"}
    elif url.rstrip("/").endswith("/api/data"):
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Fastly-Debug": "1",
            "X-Api-Type": "sports2",
            "X-SiteCode": "USAT",
        }
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=20,
            )
            response.raise_for_status()
            cache_path.write_text(response.text)
            return response.text
        except Exception as exc:  # network errors are retried with backoff
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def scrape_ratings(team: Team, end_year: int) -> list[dict[str, Any]]:
    url = (
        "https://www.hoopshype.com/nba-2k/players/"
        f"?game=nba-2k{str(end_year)[-2:]}&team={source_team_slug(team, end_year)}"
    )
    soup = BeautifulSoup(get_text(url), "html.parser")
    candidates = []
    for table in soup.select("table"):
        headers = [h.get_text(" ", strip=True) for h in table.select("thead th")]
        if "Player" not in headers or "RAT" not in headers:
            continue
        for row in table.select("tbody tr"):
            cells = row.select("td")
            link = row.select_one('a[href*="/nba-2k/players/"]')
            if len(cells) < 3 or link is None:
                continue
            rating_text = cells[-1].get_text(" ", strip=True)
            if not rating_text.isdigit():
                continue
            candidates.append(
                {
                    "name": link.get_text(" ", strip=True),
                    "rating": int(rating_text),
                    "profile_url": requests.compat.urljoin(url, link.get("href", "")),
                }
            )
        if candidates:
            break
    if len(candidates) < 10:
        raise ValueError(
            f"HoopsHype returned only {len(candidates)} players for "
            f"{team.name} {season_label(end_year)}"
        )
    return candidates


HOOPSHYPE_ALL_PLAYERS_QUERY = """
query VideoGamesOn2KAllPlayersTabPage(
    $cursor: String
    $videoGameSeries: String
    $size: Int
    $videoGameID: String
    $type: String!
) {
    videoGameRatings(
        cursor: $cursor
        videoGameSeries: $videoGameSeries
        size: $size
        videoGameID: $videoGameID
        type: $type
    ) {
        cursor
        numResults
        videoGameRatings {
            ...videoGamesOn2KAllPlayersTabPageData
        }
    }
}
fragment videoGamesOn2KAllPlayersTabPageData on VideoGameRatings {
    videoGameDisplayName
    rating
    updateDate
    sortName
    playerID
    fullPlayer {
        firstName
        lastName
        id
        team { id }
    }
}
"""


def scrape_global_ratings(end_year: int) -> list[dict[str, Any]]:
    """Fetch every player in one HoopsHype 2K edition through its page API."""
    video_game_id = f"nba-2k{str(end_year)[-2:]}"
    cursor: str | None = None
    players: list[dict[str, Any]] = []
    while True:
        variables = {
            "videoGameID": video_game_id,
            "size": 100,
            "cursor": cursor,
            "videoGameSeries": "nba-2k",
            "type": "player",
        }
        raw = get_text(
            "https://www.hoopshype.com/api/data/",
            params={
                "query": HOOPSHYPE_ALL_PLAYERS_QUERY,
                "variables": json.dumps(variables, separators=(",", ":")),
            },
        )
        block = json.loads(raw).get("data", {}).get("videoGameRatings", {})
        page = block.get("videoGameRatings", [])
        if not page:
            break
        for item in page:
            full_player = item.get("fullPlayer") or {}
            player_id = str(item.get("playerID") or full_player.get("id") or "")
            name = item.get("sortName") or " ".join(
                value
                for value in (full_player.get("firstName"), full_player.get("lastName"))
                if value
            )
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
            if not name or not player_id:
                continue
            players.append(
                {
                    "name": name,
                    "rating": int(item["rating"]),
                    "profile_url": (
                        f"https://www.hoopshype.com/nba-2k/players/{slug}/{player_id}/"
                    ),
                }
            )
        next_cursor = block.get("cursor")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    if len(players) < 300:
        raise ValueError(
            f"HoopsHype returned only {len(players)} global ratings for "
            f"{season_label(end_year)}"
        )
    return players


def opening_roster_from_wikitext(
    content: str,
    team_name: str,
    end_year: int,
) -> list[str]:
    players: list[str] = []
    in_players = False
    for line in content.splitlines():
        if re.match(r"\s*\|\s*players\s*=", line, flags=re.I):
            in_players = True
            continue
        if in_players and line.lstrip().startswith("}}"):
            break
        if not in_players or not re.match(r"\s*\*", line):
            continue
        links = re.findall(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]+)?\]\]", line)
        if links:
            players.append(re.sub(r"\s*\([^)]*\)\s*$", "", links[0]).strip())
    if len(players) < 10:
        raise ValueError(
            f"Archived roster returned only {len(players)} players for "
            f"{team_name} {season_label(end_year)}"
        )
    return players


def scrape_opening_rosters(
    teams: tuple[Team, ...],
    end_year: int,
    workers: int,
) -> dict[str, list[str]]:
    cutoff = OPENING_ROSTER_CUTOFFS.get(end_year)
    if cutoff is None:
        raise ValueError(f"No opening-roster cutoff configured for {end_year}")

    def fetch(team: Team) -> tuple[str, list[str]]:
        raw = get_text(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "revisions",
                "titles": f"Template:{team.name} current roster",
                "rvprop": "ids|timestamp|content",
                "rvslots": "main",
                "rvstart": cutoff,
                "rvdir": "older",
                "rvlimit": 1,
                "format": "json",
                "formatversion": 2,
            },
        )
        pages = json.loads(raw).get("query", {}).get("pages", [])
        revisions = pages[0].get("revisions", []) if pages else []
        if not revisions:
            raise ValueError(
                f"No archived roster revision for {team.name} {season_label(end_year)}"
            )
        content = revisions[0]["slots"]["main"]["content"]
        return team.name, opening_roster_from_wikitext(
            content, team.name, end_year
        )

    output: dict[str, list[str]] = {}
    # MediaWiki throttles bursts of historical-revision queries. Fetch these
    # sequentially and cache every successful response.
    for team in teams:
        team_name, roster = fetch(team)
        output[team_name] = roster
        time.sleep(1.0)
    missing = [team.name for team in teams if team.name not in output]
    if missing:
        raise ValueError(
            f"Missing archived opening rosters for {season_label(end_year)}: {missing}"
        )
    return output


def scrape_wikipedia_positions(team: Team, end_year: int) -> dict[str, str]:
    start = end_year - 1
    wiki_season = f"{start}\u2013{str(end_year)[-2:]}"
    title = f"{wiki_season}_{source_wiki_name(team, end_year)}_season"
    raw = get_text(f"https://en.wikipedia.org/wiki/{title}")
    soup = BeautifulSoup(raw, "html.parser")
    positions: dict[str, str] = {}
    for table in soup.select("table"):
        text = table.get_text(" ", strip=True)
        if "roster" not in text.lower() or "Pos." not in text or "Player" not in text:
            continue
        for row in table.select("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.select("th,td")]
            if len(cells) >= 3 and re.fullmatch(r"(?:PG|SG|SF|PF|C|G|F)(?:/(?:G|F|C))?", cells[0]):
                positions[normalize_name(cells[2])] = cells[0]
        if positions:
            break
    return positions


def scrape_profile_position(profile_url: str) -> str:
    soup = BeautifulSoup(get_text(profile_url), "html.parser")
    card = soup.select_one('a[href*="/salaries/teams/"]')
    if card is None:
        return "UNK"
    for value in card.stripped_strings:
        if re.fullmatch(r"(?:PG|SG|SF|PF|C|G|F)(?:/(?:G|F|C))?", value):
            return value
    return "UNK"


def scrape_standings(end_year: int) -> dict[str, dict[str, float]]:
    raw = get_text(
        "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings",
        params={"season": end_year},
    )
    payload = json.loads(raw)
    results: dict[str, dict[str, float]] = {}
    for conference in payload["children"]:
        for entry in conference["standings"]["entries"]:
            stats = {item["name"]: item.get("value") for item in entry["stats"]}
            results[entry["team"]["displayName"]] = {
                "wins": int(stats["wins"]),
                "losses": int(stats["losses"]),
                "win_pct": float(stats["winPercent"]),
            }
    return results


def scrape_espn_opening_rosters(
    end_year: int,
    workers: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, str]]]:
    cutoff = OPENING_ROSTER_CUTOFFS.get(end_year)
    if cutoff is None:
        raise ValueError(f"No opening-roster cutoff configured for {end_year}")
    first_day = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    last_day = first_day + timedelta(days=7)
    raw = get_text(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        params={
            "dates": f"{first_day:%Y%m%d}-{last_day:%Y%m%d}",
            "limit": 1000,
        },
    )
    events = sorted(json.loads(raw).get("events", []), key=lambda event: event["date"])
    event_ids_by_team: dict[str, list[str]] = {}
    for event in events:
        for competitor in event["competitions"][0]["competitors"]:
            source_name = competitor["team"]["displayName"]
            team_name = ESPN_OPENING_TEAM_ALIASES.get(source_name, source_name)
            event_ids_by_team.setdefault(team_name, []).append(str(event["id"]))
    expected_teams = {team.name for team in TEAMS}
    if set(event_ids_by_team) != expected_teams:
        raise ValueError(
            f"ESPN opening-week coverage mismatch for {season_label(end_year)}: "
            f"missing {sorted(expected_teams - set(event_ids_by_team))}, "
            f"extra {sorted(set(event_ids_by_team) - expected_teams)}"
        )

    event_ids = sorted({values[0] for values in event_ids_by_team.values()})
    summaries: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                get_text,
                "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                params={"event": event_id},
            ): event_id
            for event_id in event_ids
        }
        for future in as_completed(futures):
            summaries[futures[future]] = json.loads(future.result())

    rosters: dict[str, list[str]] = {}
    positions: dict[str, dict[str, str]] = {}
    for team_name, team_event_ids in event_ids_by_team.items():
        athlete_rows: list[dict[str, Any]] = []
        for event_id in team_event_ids:
            if event_id not in summaries:
                summaries[event_id] = json.loads(
                    get_text(
                        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary",
                        params={"event": event_id},
                    )
                )
            summary = summaries[event_id]
            team_block = next(
                (
                    block
                    for block in summary.get("boxscore", {}).get("players", [])
                    if ESPN_OPENING_TEAM_ALIASES.get(
                        block["team"]["displayName"], block["team"]["displayName"]
                    )
                    == team_name
                ),
                None,
            )
            if team_block is not None:
                athlete_rows = team_block.get("statistics", [{}])[0].get(
                    "athletes", []
                )
            if len(athlete_rows) >= 10:
                break
        rosters[team_name] = [
            athlete["athlete"]["displayName"] for athlete in athlete_rows
        ]
        positions[team_name] = {
            normalize_name(athlete["athlete"]["displayName"]): athlete["athlete"]
            .get("position", {})
            .get("abbreviation", "UNK")
            for athlete in athlete_rows
        }
        if len(rosters[team_name]) < 10:
            raise ValueError(
                f"ESPN returned only {len(rosters[team_name])} opening players for "
                f"{team_name} {season_label(end_year)}"
            )
    return rosters, positions


def build_dataset(workers: int, years: tuple[int, ...]) -> pd.DataFrame:
    ratings: dict[tuple[int, str], list[dict[str, Any]]] = {}
    positions: dict[tuple[int, str], dict[str, str]] = {}
    opening_rosters: dict[tuple[int, str], list[str]] = {}
    opening_positions: dict[tuple[int, str], dict[str, str]] = {}
    standings = {year: scrape_standings(year) for year in years}
    with ThreadPoolExecutor(max_workers=min(workers, len(years))) as pool:
        global_futures = {
            pool.submit(scrape_global_ratings, year): year for year in years
        }
        global_players = {
            global_futures[future]: future.result()
            for future in as_completed(global_futures)
        }
    for year in years:
        season_rosters, season_positions = scrape_espn_opening_rosters(year, workers)
        for team_name, players in season_rosters.items():
            opening_rosters[(year, team_name)] = players
            opening_positions[(year, team_name)] = season_positions[team_name]

    jobs = [(team, year) for year in years for team in TEAMS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rating_futures = {
            pool.submit(scrape_ratings, team, year): (team, year) for team, year in jobs
        }
        position_futures = {
            pool.submit(scrape_wikipedia_positions, team, year): (team, year)
            for team, year in jobs
        }
        for future in as_completed(rating_futures):
            team, year = rating_futures[future]
            ratings[(year, team.name)] = future.result()
        for future in as_completed(position_futures):
            team, year = position_futures[future]
            try:
                positions[(year, team.name)] = future.result()
            except Exception:
                # Position fallbacks are fetched from the corresponding HoopsHype
                # player profiles below; one unavailable roster must not discard
                # all successfully scraped ratings.
                positions[(year, team.name)] = {}

    # HoopsHype's historical team filter contains anyone associated with the
    # team during the edition, including midseason arrivals and departures.
    # Intersect those ratings with the archived opening roster, then use a
    # same-edition global index only as a name/team-page fallback.
    global_ratings: dict[int, dict[str, dict[str, Any]]] = {}
    for year, players in global_players.items():
        year_index = global_ratings.setdefault(year, {})
        for player in players:
            key = normalize_name(player["name"])
            previous = year_index.get(key)
            if previous is None or player["rating"] > previous["rating"]:
                year_index[key] = player

    selected_ratings: dict[tuple[int, str], list[dict[str, Any]]] = {}
    augmented_opening_rosters: dict[tuple[int, str], list[str]] = {}
    for key in sorted(opening_rosters):
        year, team_name = key
        official_names = opening_rosters[key]
        official_normalized = {normalize_name(value) for value in official_names}
        team_index = {
            normalize_name(player["name"]): player for player in ratings[key]
        }
        prior_names = augmented_opening_rosters.get((year - 1, team_name), [])
        returning_injured = [
            name
            for name in prior_names
            if normalize_name(name) in team_index
            and normalize_name(name) not in official_normalized
        ]
        roster_names = [*official_names, *returning_injured]
        augmented_opening_rosters[key] = roster_names
        matched: list[dict[str, Any]] = []
        seen: set[str] = set()
        for roster_name in roster_names:
            normalized = normalize_name(roster_name)
            player = team_index.get(normalized)
            if player is None and normalized in official_normalized:
                player = global_ratings[year].get(normalized)
            if player is not None and normalized not in seen:
                matched.append({**player, "rating_source": "hoopshype_edition"})
                seen.add(normalized)
        # A few fringe opening-night players were not included in older 2K
        # editions at all. Keep the opening roster intact and give those players
        # the published edition's floor rating instead of substituting a later
        # trade acquisition.
        edition_floor = min(player["rating"] for player in global_ratings[year].values())
        for roster_name in official_names:
            normalized = normalize_name(roster_name)
            if normalized in seen:
                continue
            matched.append(
                {
                    "name": roster_name,
                    "rating": edition_floor,
                    "profile_url": "",
                    "rating_source": "hoopshype_edition_floor_unrated",
                }
            )
            seen.add(normalized)
        matched.sort(key=lambda player: (-player["rating"], player["name"]))
        if len(matched) < 10:
            missing = [name for name in roster_names if normalize_name(name) not in seen]
            raise ValueError(
                f"Only {len(matched)} opening-roster ratings matched for "
                f"{team_name} {season_label(year)}; unmatched: {missing}"
            )
        selected_ratings[key] = matched[:10]

    missing_profiles: dict[str, str] = {}
    for key, players in selected_ratings.items():
        season_positions = positions.get(key, {})
        for player in players:
            if (
                player["profile_url"]
                and normalize_name(player["name"]) not in season_positions
            ):
                missing_profiles[player["profile_url"]] = player["name"]

    profile_positions: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scrape_profile_position, url): url for url in missing_profiles
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                profile_positions[url] = future.result()
            except Exception:
                profile_positions[url] = "UNK"

    rows: list[dict[str, Any]] = []
    for year in years:
        for team in TEAMS:
            record = standings[year].get(source_team_name(team, year))
            if record is None:
                raise KeyError(f"Missing ESPN standings entry for {team.name}, {year}")
            row: dict[str, Any] = {
                "team": team.name,
                "season": season_label(year),
                "season_end_year": year,
                **record,
            }
            season_positions = positions.get((year, team.name), {})
            espn_positions = opening_positions.get((year, team.name), {})
            for rank, player in enumerate(selected_ratings[(year, team.name)], start=1):
                normalized = normalize_name(player["name"])
                if normalized in espn_positions:
                    position = espn_positions[normalized]
                    source = "espn_opening_boxscore"
                elif normalized in season_positions:
                    position = season_positions[normalized]
                    source = "wikipedia_season_roster"
                else:
                    position = profile_positions.get(player["profile_url"], "UNK")
                    source = "hoopshype_profile_fallback"
                row[f"player_{rank}"] = player["name"]
                row[f"rating_{rank}"] = player["rating"]
                row[f"rating_source_{rank}"] = player["rating_source"]
                row[f"position_{rank}"] = position
                row[f"position_source_{rank}"] = source
            row["roster_snapshot"] = "opening_night"
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["season_end_year", "team"]).reset_index(drop=True)
    expected_rows = 30 * len(years)
    if len(frame) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} team-seasons, got {len(frame)}")
    rating_cols = [f"rating_{i}" for i in range(1, 11)]
    if frame[rating_cols].isna().any().any():
        raise AssertionError("Missing one or more top-10 ratings")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/nba_2k_team_seasons.csv")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start-end-year", type=int, default=DEFAULT_START_END_YEAR)
    parser.add_argument("--end-end-year", type=int, default=DEFAULT_END_END_YEAR)
    args = parser.parse_args()
    if args.start_end_year > args.end_end_year:
        parser.error("--start-end-year must be <= --end-end-year")
    years = tuple(range(args.start_end_year, args.end_end_year + 1))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(max(1, args.workers), years)
    dataset.to_csv(output, index=False)
    fallback_cols = [f"position_source_{i}" for i in range(1, 11)]
    fallback_count = int((dataset[fallback_cols] == "hoopshype_profile_fallback").sum().sum())
    print(f"Wrote {len(dataset)} rows x {len(dataset.columns)} columns to {output}")
    total_slots = len(dataset) * 10
    print(f"Season-roster positions: {total_slots - fallback_count}; profile fallbacks: {fallback_count}")


if __name__ == "__main__":
    main()
