export type ZoneStat = {
  key:
    | "rim"
    | "paint"
    | "midrange"
    | "left_corner_3"
    | "right_corner_3"
    | "above_break_3";
  label: string;
  made: number;
  attempts: number;
  pct: number | null;
  leaguePct: number | null;
  delta: number | null;
  frequency: number;
};

export type MatchupDefense = {
  possessions: number;
  minutes: number;
  points: number;
  fgm: number;
  fga: number;
  fgPct: number | null;
  twoPm: number;
  twoPa: number;
  twoPct: number | null;
  threePm: number;
  threePa: number;
  threePct: number | null;
  turnovers: number;
  blocks: number;
};

export type ScoutPlayer = {
  id: string;
  name: string;
  jersey: string | null;
  position: string;
  positions: string[];
  rating: number;
  headshotUrl: string | null;
  headshotVerified: boolean;
  status: string;
  offense: {
    attempts: number;
    made: number;
    zones: ZoneStat[];
  };
  defense: MatchupDefense;
};

export type ScoutTeam = {
  id: string;
  name: string;
  shortName: string;
  abbreviation: string;
  slug: string;
  logoUrl: string;
  color: string;
  players: ScoutPlayer[];
  projected: {
    starters: string[];
    secondUnit: string[];
  };
  defenseZones: ZoneStat[];
};

export type ScoutingData = {
  metadata: {
    statsSeason: string;
    rosterSeason: string;
    sourceRosterGeneratedAt: string;
    generatedAt: string;
    rosterAuthority: string;
    headshotAuthority: string;
    ratingsSource: string;
    depthChartSource: string;
    rotationMethod: string;
    shotSource: string;
    matchupSource: string;
    teamCount: number;
    playerCount: number;
    playersWithOffense: number;
    playersWithDefense: number;
    skippedTeamShotRows: number;
  };
  teams: ScoutTeam[];
};
