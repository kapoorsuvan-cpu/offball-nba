import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dashboard-app/app/scout/data/scouting-data.json"


class ScoutingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(DATA_PATH.read_text())
        cls.teams = {team["abbreviation"]: team for team in cls.data["teams"]}

    def unit_names(self, abbreviation: str, unit: str) -> list[str]:
        team = self.teams[abbreviation]
        by_id = {player["id"]: player for player in team["players"]}
        return [by_id[player_id]["name"] for player_id in team["projected"][unit]]

    def player(self, abbreviation: str, name: str) -> dict:
        return next(
            player for player in self.teams[abbreviation]["players"] if player["name"] == name
        )

    def test_all_teams_have_ten_unique_available_players(self) -> None:
        self.assertEqual(len(self.teams), 30)
        for abbreviation, team in self.teams.items():
            by_id = {player["id"]: player for player in team["players"]}
            rotation = team["projected"]["starters"] + team["projected"]["secondUnit"]
            self.assertEqual(len(rotation), 10, abbreviation)
            self.assertEqual(len(set(rotation)), 10, abbreviation)
            self.assertTrue(all(by_id[player_id]["status"] == "Active" for player_id in rotation))
            starters = [by_id[player_id] for player_id in team["projected"]["starters"]]
            self.assertTrue(any("C" in player["positions"] for player in starters), abbreviation)

    def test_named_depth_chart_decisions(self) -> None:
        self.assertIn("James Harden", self.unit_names("CLE", "starters"))
        self.assertIn("Matisse Thybulle", self.unit_names("LAL", "secondUnit"))
        self.assertNotIn("Cameron Carr", self.unit_names("LAL", "secondUnit"))
        self.assertIn("AJ Dybantsa", self.unit_names("WSH", "starters"))
        self.assertIn("Cameron Boozer", self.unit_names("MEM", "starters"))
        self.assertNotIn("Jarace Walker", self.unit_names("IND", "starters"))
        self.assertIn("Kawhi Leonard", self.unit_names("TOR", "starters"))

    def test_added_players_and_ratings(self) -> None:
        self.assertIn("Nick Richards", self.unit_names("MIA", "secondUnit"))
        self.assertEqual(self.player("MIA", "Nick Richards")["rating"], 75)
        self.assertIn("Haywood Highsmith", self.unit_names("PHX", "secondUnit"))
        self.assertEqual(self.player("PHX", "Haywood Highsmith")["rating"], 73)
        atlanta = {player["name"]: player["rating"] for player in self.teams["ATL"]["players"]}
        self.assertEqual(atlanta["Jalen Johnson"], 87)
        self.assertEqual(atlanta["Dyson Daniels"], 81)
        self.assertEqual(atlanta["Onyeka Okongwu"], 81)

    def test_jimmy_butler_is_unavailable(self) -> None:
        jimmy = self.player("GS", "Jimmy Butler III")
        self.assertTrue(jimmy["status"].startswith("Out"))
        warriors_rotation = self.unit_names("GS", "starters") + self.unit_names(
            "GS", "secondUnit"
        )
        self.assertNotIn("Jimmy Butler III", warriors_rotation)

    def test_payton_watson_trade_roster_moves(self) -> None:
        cleveland = {player["name"] for player in self.teams["CLE"]["players"]}
        clippers = {player["name"] for player in self.teams["LAC"]["players"]}
        wizards = {player["name"] for player in self.teams["WSH"]["players"]}
        hornets = {player["name"] for player in self.teams["CHA"]["players"]}
        nuggets = {player["name"] for player in self.teams["DEN"]["players"]}
        self.assertTrue({"Peyton Watson", "Cam Whitmore"}.issubset(cleveland))
        self.assertIn("Max Strus", clippers)
        self.assertIn("Tre Mann", wizards)
        self.assertIn("Dennis Schroder", hornets)
        self.assertNotIn("Julian Reese", nuggets)

    def test_source_metadata(self) -> None:
        metadata = self.data["metadata"]
        self.assertEqual(metadata["teamCount"], 30)
        self.assertEqual(metadata["depthChartSource"], "https://www.nbadepthcharts.com")
        self.assertEqual(metadata["ratingsSource"], "https://www.2kratings.com/teams")
        self.assertIn("espn.com", metadata["playerStatsSource"])
        self.assertEqual(
            metadata["advancedStatsSource"],
            "https://www.basketball-reference.com/leagues/NBA_2026_advanced.html",
        )

    def test_keyonte_george_stats_and_headshot_fallback(self) -> None:
        keyonte = self.player("UTAH", "Keyonte George")
        stats = keyonte["seasonStats"]
        self.assertEqual(stats["games"], 54)
        self.assertAlmostEqual(stats["points"], 23.6)
        self.assertAlmostEqual(stats["assists"], 6.1)
        self.assertAlmostEqual(stats["tsPct"], 60.9)
        self.assertAlmostEqual(stats["bpm"], 0.9)
        self.assertTrue(keyonte["headshotFallbackUrl"].endswith("/1641718.png"))


if __name__ == "__main__":
    unittest.main()
