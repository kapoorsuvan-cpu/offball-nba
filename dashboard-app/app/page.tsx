import type { Metadata } from "next";
import rosterData from "./data/current-rosters.json";
import { Dashboard, type DashboardData } from "./dashboard";

export const metadata: Metadata = {
  title: "OFFBALL · NBA Win Predictor",
  description: "Explore current NBA rosters, spread-calibrated Lasso win forecasts, and ten seasons of walk-forward results.",
};

export default function Home() {
  return <Dashboard data={rosterData as DashboardData} />;
}
