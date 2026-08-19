import type { Metadata } from "next";
import scoutingData from "./data/scouting-data.json";
import { ScoutingDashboard } from "./scouting-dashboard";
import type { ScoutingData } from "./types";

export const metadata: Metadata = {
  title: "NBA Scouting Lab",
  description:
    "Player shot profiles, team defensive concessions, current rosters, and projected ten-man rotations for all 30 NBA teams.",
  openGraph: {
    title: "OFFBALL NBA Scouting Lab",
    description:
      "Shot-zone profiles, defensive concessions, headshots, and projected rotations for all 30 teams.",
    images: [{ url: "/scout-og.png", width: 1600, height: 1000 }],
  },
  twitter: {
    card: "summary_large_image",
    images: ["/scout-og.png"],
  },
};

export default function ScoutPage() {
  return <ScoutingDashboard data={scoutingData as ScoutingData} />;
}
