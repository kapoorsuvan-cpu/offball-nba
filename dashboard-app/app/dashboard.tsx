"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

type Player = {
  id: string;
  name: string;
  jersey: string | null;
  position: string;
  positions: string[];
  positionDisplay: string;
  positionSource: string;
  rating: number;
  ratingSource: string;
  ratingSourceUrl: string | null;
  ratingPageTeam: string | null;
  headshotUrl: string | null;
  headshotVerified: boolean;
  espnUrl: string | null;
  age: number | null;
  height: string | null;
  weight: string | null;
  status: string;
  experienceYears: number | null;
  modelRank: number | null;
  ratingSensitivity: number;
  impactWins: number;
};

type Team = {
  id: string;
  name: string;
  shortName: string;
  abbreviation: string;
  slug: string;
  logoUrl: string;
  color: string;
  espnRosterUrl: string;
  rosterCount: number;
  players: Player[];
  topTen: string[];
  positionMix: Record<"PG" | "SG" | "SF" | "PF" | "C", number>;
  prediction: {
    wins: number;
    losses: number;
    record: string;
    scheduleGames: number;
    confidenceLow: number;
    confidenceHigh: number;
    distribution: { wins: number; probability: number }[];
    model: string;
    scenarioSensitivities: {
      starOne: number;
      starTwo: number;
      starterCore: number;
      benchDepth: number;
    } | null;
  };
  actual?: {
    wins: number;
    losses: number;
    record: string;
    gamesPlayed: number;
    winPct: number;
    normalizedWins: number;
  };
};

type HistoricalSeason = {
  season: string;
  seasonEndYear: number;
  teams: Team[];
};

type ModelMetric = {
  id: string;
  name: string;
  maeWins: number;
  rmseWins: number;
  r2: number;
  cvRmseWins: number | null;
};

type BacktestYear = {
  season: string;
  bets: number;
  wins: number;
  losses: number;
  voids: number;
  netProfit: number;
  modelMaeWins: number;
  marketMaeWins: number;
};

export type DashboardData = {
  metadata: {
    season: string;
    generatedAt: string;
    generatedLocalDate: string;
    rosterAuthority: string;
    headshotAuthority: string;
    ratingAuthority: string;
    fallbackPolicy: string;
    model: string;
    modelSelection: string;
    selectedFeatureCount: number;
    selectedFeatures: { feature: string; coefficient_wins: number }[];
    heldoutMaeWins: number;
    heldoutRmseWins: number;
    heldoutR2: number;
    trainingMaeWins: number;
    trainingRmseWins: number;
    validationMaeWins: number;
    validationRmseWins: number;
    trainValidationRmseGapWins: number;
    currentPredictionRangeWins: number;
    previousPredictionRangeWins: number | null;
    rankCalibrationBlend: number;
    minimumValidationSpreadRatio: number;
    validationSpreadRatio: number;
    heldoutSpreadRatio: number;
    validationTailBias: {
      under30BiasWins: number;
      fiftyPlusBiasWins: number;
    };
    heldoutTailBias: {
      under30BiasWins: number;
      fiftyPlusBiasWins: number;
    };
    sources: { label: string; url: string }[];
  };
  audit: {
    teamCount: number;
    playerCount: number;
    headshotUrlCount: number;
    verifiedHeadshotCount: number;
    matchCounts: Record<string, number>;
    positionSourceCounts: Record<string, number>;
    invalidPositionPlayers: { player: string; team: string; positions: string[] }[];
    crossTeamRatingMatches: { player: string; espnTeam: string; ratingPageTeam: string }[];
    unmatchedPlayers: { player: string; team: string }[];
    officialMoveChecks: { player: string; team: string; status: string }[];
  };
  modelMetrics: ModelMetric[];
  backtest: {
    years: BacktestYear[];
    overall: {
      bets: number;
      settledBets: number;
      wins: number;
      losses: number;
      voids: number;
      winRate: number;
      netProfit: number;
      roi: number;
      modelMaeWins: number;
      marketMaeWins: number;
      pValue: number;
    };
  };
  teams: Team[];
  historicalSeasons: HistoricalSeason[];
};

type IconName =
  | "dashboard"
  | "teams"
  | "players"
  | "models"
  | "backtest"
  | "search"
  | "chevron"
  | "verified"
  | "info"
  | "arrowUp"
  | "arrowDown"
  | "external";

const iconPaths: Record<IconName, React.ReactNode> = {
  dashboard: <><path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10.5V20h13v-9.5"/></>,
  teams: <><circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3.5 20v-2.5A4.5 4.5 0 0 1 8 13h2a4.5 4.5 0 0 1 4.5 4.5V20"/><path d="M14.5 14.5a4 4 0 0 1 6 3.5v2"/></>,
  players: <><circle cx="12" cy="7" r="3.5"/><path d="M5 21v-3a7 7 0 0 1 14 0v3"/></>,
  models: <><circle cx="5" cy="18" r="2.5"/><circle cx="12" cy="6" r="2.5"/><circle cx="19" cy="18" r="2.5"/><path d="m6.3 15.9 4.4-7.8m2.6 0 4.4 7.8M7.5 18h9"/></>,
  backtest: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3.5 2"/></>,
  search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4 4"/></>,
  chevron: <path d="m8 10 4 4 4-4"/>,
  verified: <><circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16.5 9"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5m0-8v.1"/></>,
  arrowUp: <><path d="m7 12 5-5 5 5"/><path d="M12 7v10"/></>,
  arrowDown: <><path d="m7 12 5 5 5-5"/><path d="M12 17V7"/></>,
  external: <><path d="M14 5h5v5"/><path d="m19 5-8 8"/><path d="M18 13v6H5V6h6"/></>,
};

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {iconPaths[name]}
    </svg>
  );
}

function PlayerAvatar({ player, size = 36 }: { player: Player; size?: number }) {
  const [failed, setFailed] = useState(false);
  const initials = player.name.split(" ").slice(0, 2).map((part) => part[0]).join("");
  return (
    <span className="avatar" style={{ width: size, height: size }}>
      {failed || !player.headshotUrl ? (
        <span className="avatar-fallback">{initials}</span>
      ) : (
        <Image
          src={player.headshotUrl}
          alt={`${player.name} headshot`}
          width={size}
          height={size}
          sizes={`${size}px`}
          unoptimized
          onError={() => setFailed(true)}
        />
      )}
    </span>
  );
}

function TeamLogo({ team, size = 34 }: { team: Team; size?: number }) {
  return <Image src={team.logoUrl} alt={`${team.name} logo`} width={size} height={size} sizes={`${size}px`} unoptimized />;
}

function ModelIcon({ id }: { id: string }) {
  if (id.includes("extra_trees")) return <span className="model-symbol">⌁</span>;
  if (id.startsWith("tabfm")) return <Icon name="models" size={28} />;
  return <span className="model-symbol">⎇</span>;
}

function BacktestChart({ years }: { years: BacktestYear[] }) {
  const width = 720;
  const height = 260;
  const padding = { left: 36, right: 14, top: 18, bottom: 34 };
  const maxY = 14;
  const minY = 3;
  const x = (index: number) => padding.left + (index / (years.length - 1)) * (width - padding.left - padding.right);
  const y = (value: number) => padding.top + ((maxY - value) / (maxY - minY)) * (height - padding.top - padding.bottom);
  const points = (key: "modelMaeWins" | "marketMaeWins") => years.map((year, index) => `${x(index)},${y(year[key])}`).join(" ");
  return (
    <div className="chart-wrap">
      <svg className="backtest-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Model and market mean absolute error by season">
        {[4, 8, 12].map((value) => (
          <g key={value}>
            <line x1={padding.left} y1={y(value)} x2={width - padding.right} y2={y(value)} className="grid-line" />
            <text x={8} y={y(value) + 4} className="axis-label">{value}</text>
          </g>
        ))}
        <polyline points={points("marketMaeWins")} className="market-line" />
        <polyline points={points("modelMaeWins")} className="model-line" />
        {years.map((year, index) => index % 2 === 0 || index === years.length - 1 ? (
          <text key={year.season} x={x(index)} y={height - 8} textAnchor="middle" className="axis-label">{year.season.slice(0, 4)}</text>
        ) : null)}
      </svg>
    </div>
  );
}

function money(value: number) {
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}$${Math.abs(value).toFixed(2)}`;
}

export function Dashboard({ data }: { data: DashboardData }) {
  const initialTeam = data.teams.find((team) => team.abbreviation === "BOS") ?? data.teams[0];
  const [selectedSeason, setSelectedSeason] = useState(data.metadata.season);
  const [selectedTeamAbbreviation, setSelectedTeamAbbreviation] = useState(initialTeam.abbreviation);
  const [position, setPosition] = useState("ALL");
  const [ratingBand, setRatingBand] = useState("ALL");
  const [availability, setAvailability] = useState("ALL");
  const [sortBy, setSortBy] = useState("rating");
  const [teamPickerOpen, setTeamPickerOpen] = useState(false);
  const [swingInfoOpen, setSwingInfoOpen] = useState(false);
  const swingCloseRef = useRef<HTMLButtonElement>(null);
  const [query, setQuery] = useState("");
  const [injuryRisk, setInjuryRisk] = useState(false);
  const [starOneRating, setStarOneRating] = useState(initialTeam.players[0].rating);
  const [starTwoRating, setStarTwoRating] = useState(initialTeam.players[1].rating);
  const initialStarterCore = Math.round(initialTeam.players.slice(2, 5).reduce((sum, player) => sum + player.rating, 0) / 3);
  const initialBenchDepth = Math.round(initialTeam.players.slice(5, 10).reduce((sum, player) => sum + player.rating, 0) / 5);
  const [starterCoreScore, setStarterCoreScore] = useState(initialStarterCore);
  const [benchDepthScore, setBenchDepthScore] = useState(initialBenchDepth);

  const activeHistoricalSeason = data.historicalSeasons.find((season) => season.season === selectedSeason);
  const isCurrentSeason = !activeHistoricalSeason;
  const seasonTeams = activeHistoricalSeason?.teams ?? data.teams;
  const team = seasonTeams.find((item) => item.abbreviation === selectedTeamAbbreviation) ?? seasonTeams[0];
  const featuredAbbreviations = ["BOS", "LAL", "OKC", "DEN", "NY"];
  const featuredTeams = featuredAbbreviations.map((abbr) => seasonTeams.find((item) => item.abbreviation === abbr)).filter((item): item is Team => Boolean(item));

  const resetScenario = (nextTeam: Team) => {
    setStarOneRating(nextTeam.players[0].rating);
    setStarTwoRating(nextTeam.players[1].rating);
    const starters = nextTeam.players.slice(2, 5);
    const bench = nextTeam.players.slice(5, 10);
    setStarterCoreScore(Math.round(starters.reduce((sum, player) => sum + player.rating, 0) / Math.max(starters.length, 1)));
    setBenchDepthScore(Math.round(bench.reduce((sum, player) => sum + player.rating, 0) / Math.max(bench.length, 1)));
    setInjuryRisk(false);
  };

  const chooseTeam = (nextTeam: Team) => {
    setSelectedTeamAbbreviation(nextTeam.abbreviation);
    resetScenario(nextTeam);
    setPosition("ALL");
    setRatingBand("ALL");
    setAvailability("ALL");
    setTeamPickerOpen(false);
    setQuery("");
  };

  const chooseSeason = (season: string) => {
    const historical = data.historicalSeasons.find((item) => item.season === season);
    const nextTeams = historical?.teams ?? data.teams;
    const nextTeam = nextTeams.find((item) => item.abbreviation === selectedTeamAbbreviation) ?? nextTeams[0];
    setSelectedSeason(season);
    chooseTeam(nextTeam);
  };

  const filteredPlayers = useMemo(() => {
    const minimum = ratingBand === "80" ? 80 : ratingBand === "75" ? 75 : 0;
    const maximum = ratingBand === "UNDER75" ? 74 : 99;
    return team.players
      .filter((player) => position === "ALL" || player.positions.includes(position))
      .filter((player) => player.rating >= minimum && player.rating <= maximum)
      .filter((player) => availability === "ALL" || player.status.toUpperCase() === availability)
      .toSorted((a, b) => {
        if (sortBy === "name") return a.name.localeCompare(b.name);
        if (sortBy === "position") return a.position.localeCompare(b.position) || b.rating - a.rating;
        return b.rating - a.rating || a.name.localeCompare(b.name);
      });
  }, [availability, position, ratingBand, sortBy, team.players]);

  const searchResults = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (normalized.length < 2) return [];
    const results: { key: string; label: string; detail: string; team: Team; player?: Player }[] = [];
    for (const candidate of seasonTeams) {
      if (candidate.name.toLowerCase().includes(normalized) || candidate.abbreviation.toLowerCase().includes(normalized)) {
        results.push({ key: `team-${candidate.id}`, label: candidate.name, detail: "Team", team: candidate });
      }
      for (const player of candidate.players) {
        if (player.name.toLowerCase().includes(normalized)) {
          results.push({ key: `player-${player.id}`, label: player.name, detail: `${candidate.abbreviation} · ${player.position} · ${player.rating} OVR`, team: candidate, player });
        }
      }
    }
    return results.slice(0, 8);
  }, [query, seasonTeams]);

  const starterPlayers = team.players.slice(2, 5);
  const baseStarterCore = Math.round(starterPlayers.reduce((sum, player) => sum + player.rating, 0) / Math.max(starterPlayers.length, 1));
  const sensitivity = team.prediction.scenarioSensitivities;
  const scenarioDelta = isCurrentSeason && sensitivity ?
    (starOneRating - team.players[0].rating) * sensitivity.starOne +
    (starTwoRating - team.players[1].rating) * sensitivity.starTwo +
    (starterCoreScore - baseStarterCore) * sensitivity.starterCore +
    (injuryRisk ? -1.6 : 0) : 0;
  const predictionScheduleGames = team.prediction.scheduleGames ?? 82;
  const scenarioWins = Math.min(predictionScheduleGames - 8, Math.max(8, team.prediction.wins + scenarioDelta));
  const scenarioShift = scenarioWins - team.prediction.wins;
  const actualComparisonWins = team.actual ? team.actual.winPct * predictionScheduleGames : scenarioWins;
  const scheduleAdjustedPrediction = predictionScheduleGames < 82;
  const interruptedSeason = Boolean(team.actual && team.actual.gamesPlayed !== predictionScheduleGames);
  const historicalError = actualComparisonWins - scenarioWins;
  const topMetrics = ["rank_calibrated_lasso", "extra_trees", "tabfm_jax"].map((id) => data.modelMetrics.find((metric) => metric.id === id)).filter((metric): metric is ModelMetric => Boolean(metric));
  const maxDistribution = Math.max(...team.prediction.distribution.map((bucket) => bucket.probability));
  const generatedDate = data.metadata.generatedLocalDate;
  const topImpact = team.players.slice(0, 10).toSorted((a, b) => b.impactWins - a.impactWins).slice(0, 2);
  const seasonOptions = [data.metadata.season, ...data.historicalSeasons.toReversed().map((season) => season.season)];

  useEffect(() => {
    if (!swingInfoOpen) return;
    swingCloseRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSwingInfoOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [swingInfoOpen]);

  return (
    <main className="app-shell" style={{ "--team-color": team.color } as React.CSSProperties}>
      <section className="workspace" id="top">
        <header className="topbar">
          <a className="brand" href="#top" aria-label="OFFBALL home"><Image className="brand-logo" src="/offball-logo.svg" alt="OFFBALL" width={269} height={88} priority /></a>
          <div className="topbar-controls">
            <label className="season-button">
              <span className="sr-only">Season</span>
              <select value={selectedSeason} onChange={(event) => chooseSeason(event.target.value)}>
                {seasonOptions.map((season) => <option key={season} value={season}>{season} Season</option>)}
              </select>
              <Icon name="chevron" size={17} />
            </label>
            <div className="global-search">
              <Icon name="search" size={20} />
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search teams or players" aria-label="Search teams or players" />
              {searchResults.length > 0 ? (
                <div className="search-results">
                  {searchResults.map((result) => (
                    <button key={result.key} type="button" onClick={() => chooseTeam(result.team)}>
                      {result.player ? <PlayerAvatar player={result.player} size={34} /> : <TeamLogo team={result.team} size={30} />}
                      <span><b>{result.label}</b><small>{result.detail}</small></span>
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        </header>

        <div className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="live-dot" /> {selectedSeason} {isCurrentSeason ? "ROSTER SNAPSHOT" : "BACKTEST SNAPSHOT"}</div>
              <h1>NBA Win Predictor <span>with 2K Ratings</span></h1>
            </div>
            <div className="source-note"><Icon name="verified" size={18} /><span>{isCurrentSeason ? "ESPN rosters & headshots" : "Saved historical roster matrix"}<br /><small>{isCurrentSeason ? `2K27 ratings · Updated ${generatedDate}` : `${selectedSeason} walk-forward forecast · no future seasons used`}</small></span></div>
          </div>

          <section className="team-strip" id="team-picker" aria-label="Featured teams">
            {featuredTeams.map((featured) => (
              <button key={featured.id} className={featured.id === team.id ? "selected" : ""} type="button" onClick={() => chooseTeam(featured)}>
                <TeamLogo team={featured} size={34} /><span>{featured.name}</span>
              </button>
            ))}
            <button className="all-teams-button" type="button" onClick={() => setTeamPickerOpen((open) => !open)}>
              All 30 <Icon name="chevron" size={16} />
            </button>
            {teamPickerOpen ? (
              <div className="team-picker-grid">
                {seasonTeams.toSorted((a, b) => a.name.localeCompare(b.name)).map((candidate) => (
                  <button key={candidate.id} type="button" onClick={() => chooseTeam(candidate)}>
                    <TeamLogo team={candidate} size={28} /><span>{candidate.name}</span><small>{candidate.prediction.wins.toFixed(1)} W</small>
                  </button>
                ))}
              </div>
            ) : null}
          </section>

          <div className="dashboard-grid">
            <section className="panel roster-panel" id="roster">
              <div className="panel-header roster-header">
                <div>
                  <h2>Roster Intelligence</h2>
                  <span className="roster-count">{isCurrentSeason ? `${team.rosterCount} current players` : "Top-10 saved matrix"}</span>
                </div>
                <div className="filters">
                  <label><span>Position</span><select value={position} onChange={(event) => setPosition(event.target.value)}><option value="ALL">All</option><option>PG</option><option>SG</option><option>SF</option><option>PF</option><option>C</option></select></label>
                  <label><span>Rating</span><select value={ratingBand} onChange={(event) => setRatingBand(event.target.value)}><option value="ALL">All ratings</option><option value="80">80+</option><option value="75">75+</option><option value="UNDER75">Under 75</option></select></label>
                  <label><span>Availability</span><select value={availability} onChange={(event) => setAvailability(event.target.value)}><option value="ALL">All</option><option value="ACTIVE">Active</option></select></label>
                  <label><span>Sort by</span><select value={sortBy} onChange={(event) => setSortBy(event.target.value)}><option value="rating">2K Rating</option><option value="name">Player</option><option value="position">Position</option></select></label>
                </div>
              </div>
              <div className="roster-table" role="region" aria-label={`${team.name} roster`}>
                <div className="roster-row roster-columns">
                  <span>#</span><span>Player</span><span>Pos</span><span>2K rating</span><span>3-pt swing <button className="info-button" type="button" aria-label="Explain 3-point swing" aria-haspopup="dialog" aria-expanded={swingInfoOpen} aria-controls="swing-info-dialog" onClick={() => setSwingInfoOpen(true)}><Icon name="info" size={14} /></button></span>
                </div>
                <div className="roster-scroll">
                  {filteredPlayers.map((player, index) => {
                    const maxImpact = Math.max(1, ...team.players.slice(0, 10).map((item) => item.impactWins));
                    return (
                      <a className="roster-row player-row" key={player.id} href={player.espnUrl ?? player.ratingSourceUrl ?? team.espnRosterUrl} target="_blank" rel="noreferrer">
                        <span className="rank">{index + 1}</span>
                        <span className="player-cell"><PlayerAvatar player={player} /><span><b>{player.name}</b><small>{isCurrentSeason ? `${player.height ?? "—"} · ${player.status}` : player.ratingSource}{isCurrentSeason && player.ratingSource !== "NBA 2K27 current" ? ` · ${player.ratingSource}` : ""}</small></span></span>
                        <span><span className="position-pill" title={`Position source: ${player.positionSource}`}>{player.positionDisplay}</span></span>
                        <span className="rating-cell"><b>{player.rating}</b><small>OVR</small></span>
                        <span className="impact-cell"><span className="impact-track"><i style={{ width: `${isCurrentSeason ? Math.max(4, (player.impactWins / maxImpact) * 100) : 0}%` }} /></span><em>{isCurrentSeason && player.modelRank ? `${player.impactWins >= 0 ? "+" : ""}${player.impactWins.toFixed(1)}` : "—"}</em></span>
                      </a>
                    );
                  })}
                  {filteredPlayers.length === 0 ? <div className="empty-state">No players match these filters.</div> : null}
                </div>
              </div>
            </section>

            <section className="panel projection-panel">
              <div className="panel-header"><h2>Projected Wins</h2><span className="model-badge">{team.prediction.model}</span></div>
              <div className="projection-summary">
                <div className="big-number"><strong>{scenarioWins.toFixed(1)}</strong><span>Projected wins</span></div>
                <div className="record-block"><small>{scheduleAdjustedPrediction ? `Projected ${predictionScheduleGames}-game record` : interruptedSeason ? "Projected 82-game pace" : "Projected record"}</small><b>{Math.round(scenarioWins)}–{predictionScheduleGames - Math.round(scenarioWins)}</b>{team.actual ? <><small>Actual record{team.actual.gamesPlayed !== 82 ? ` (${team.actual.gamesPlayed} games)` : ""}</small><b>{team.actual.record}</b></> : <><small>80% model band</small><b>{Math.max(0, Math.round(team.prediction.confidenceLow + scenarioShift))}–{Math.min(predictionScheduleGames, Math.round(team.prediction.confidenceHigh + scenarioShift))} <em>wins</em></b></>}</div>
              </div>
              <div className="distribution">
                <small>WIN DISTRIBUTION</small>
                <div className="distribution-bars">
                  {team.prediction.distribution.map((bucket, index) => {
                    const highlighted = Math.abs(bucket.wins - scenarioWins) < 3;
                    const tooltipEdge = index === 0 ? " first" : index === team.prediction.distribution.length - 1 ? " last" : "";
                    const probabilityLabel = `${(bucket.probability * 100).toFixed(1)}% near ${bucket.wins} wins`;
                    return <span key={bucket.wins} className={`${highlighted ? "highlighted" : ""}${tooltipEdge}`} role="img" aria-label={probabilityLabel} style={{ height: `${Math.max(5, (bucket.probability / maxDistribution) * 100)}%` }}><em className="bar-tooltip">{probabilityLabel}</em><i>{bucket.wins}</i></span>;
                  })}
                </div>
              </div>
              <div className="projection-footer"><span>{isCurrentSeason ? "Scenario change" : scheduleAdjustedPrediction ? `Forecast error (${predictionScheduleGames}-game schedule)` : interruptedSeason ? "Forecast error (82-game pace)" : "Forecast error"}</span><b className={(isCurrentSeason ? scenarioDelta : historicalError) >= 0 ? "positive" : "negative"}>{isCurrentSeason ? `${scenarioDelta >= 0 ? "+" : ""}${scenarioDelta.toFixed(1)} wins` : `${historicalError >= 0 ? "+" : ""}${historicalError.toFixed(1)} wins`}</b></div>
            </section>

            <section className="panel custom-panel">
              <div className="panel-header"><h2>{isCurrentSeason ? "Custom Prediction Inputs" : "Historical Forecast Check"}</h2>{isCurrentSeason ? <button type="button" className="reset-button" onClick={() => resetScenario(team)}>Reset</button> : null}</div>
              {isCurrentSeason ? <div className="custom-grid">
                <div className="sliders four-sliders">
                  <label><span>{team.players[0].name} rating</span><input aria-label={`${team.players[0].name} rating`} type="range" min="60" max="99" value={starOneRating} onChange={(event) => setStarOneRating(Number(event.target.value))} /><b>{starOneRating}</b></label>
                  <label><span>{team.players[1].name} rating</span><input aria-label={`${team.players[1].name} rating`} type="range" min="60" max="99" value={starTwoRating} onChange={(event) => setStarTwoRating(Number(event.target.value))} /><b>{starTwoRating}</b></label>
                  <label><span>Starter core (3–5)</span><input aria-label="Starter core average rating" type="range" min="60" max="95" value={starterCoreScore} onChange={(event) => setStarterCoreScore(Number(event.target.value))} /><b>{starterCoreScore}</b></label>
                  <label className="removed-feature"><span>Bench depth (Lasso: 0)</span><input aria-label="Bench depth removed by Lasso" type="range" min="60" max="90" value={benchDepthScore} disabled /><b>{benchDepthScore}</b></label>
                </div>
                <div className="mix-controls">
                  <span className="control-label">Top-10 position mix</span>
                  <div className="position-mix">{(["PG", "SG", "SF", "PF", "C"] as const).map((pos) => <span key={pos}>{pos} <b>{team.positionMix[pos]}</b></span>)}</div>
                  <span className="control-label">Injury-risk scenario <Icon name="info" size={14} /></span>
                  <div className="toggle-group"><button type="button" className={!injuryRisk ? "active" : ""} onClick={() => setInjuryRisk(false)}>Off</button><button type="button" className={injuryRisk ? "active" : ""} onClick={() => setInjuryRisk(true)}>On</button></div>
                </div>
              </div> : <div className="historical-summary"><div><small>Walk-forward prediction{scheduleAdjustedPrediction ? ` · ${predictionScheduleGames}-game schedule` : interruptedSeason ? " · 82-game pace" : ""}</small><b>{team.prediction.wins.toFixed(1)} wins</b></div><div><small>Actual finish{team.actual?.gamesPlayed !== 82 ? ` · ${team.actual?.gamesPlayed} games` : ""}</small><b>{team.actual?.record}{interruptedSeason ? ` · ${actualComparisonWins.toFixed(1)}-win pace` : ""}</b></div><p>This forecast was trained only on seasons before {selectedSeason}. {scheduleAdjustedPrediction ? `Prediction, confidence, distribution, and error are converted from model win percentage to the ${predictionScheduleGames}-game schedule. ` : interruptedSeason ? "Accuracy uses win percentage on an 82-game pace; 2019-20 wagers remain void. " : ""}Custom sliders are intentionally locked for historical rows so the backtest stays unchanged.</p></div>}
            </section>

            <section className="panel drivers-panel">
              <div className="panel-header"><h2>What changes the prediction?</h2></div>
              {isCurrentSeason ? <div className="drivers-list">
                {topImpact.map((player) => <div key={player.id}><span className="driver-icon up"><Icon name="arrowUp" size={16} /></span><span><b>{player.name}</b><small>Local ±3 rating scenario</small></span><strong>±{player.impactWins.toFixed(1)} wins</strong></div>)}
                <div><span className="driver-icon up"><Icon name="arrowUp" size={16} /></span><span><b>Starter core (3–5)</b><small>Learned group sensitivity</small></span><strong>+{sensitivity?.starterCore.toFixed(2)} / point</strong></div>
                <div><span className="driver-icon neutral">0</span><span><b>Bench depth (6–10)</b><small>Coefficient shrank to zero</small></span><strong>Removed by Lasso</strong></div>
                <div><span className="driver-icon down"><Icon name="arrowDown" size={16} /></span><span><b>Injury risk</b><small>Explicit scenario adjustment</small></span><strong>−1.6 wins</strong></div>
              </div> : <div className="drivers-list historical-drivers"><div><span className="driver-icon up"><Icon name="verified" size={16} /></span><span><b>Strict walk-forward training</b><small>No future season outcomes used</small></span><strong>{selectedSeason}</strong></div><div><span className="driver-icon up"><Icon name="verified" size={16} /></span><span><b>Saved inputs</b><small>Top 10 ratings and year-specific positions</small></span><strong>10 players</strong></div></div>}
              <p className="driver-note">{isCurrentSeason ? "All rating controls use local sensitivity from the trained Lasso model. Injury risk is a transparent scenario adjustment, not a trained feature." : "Switch teams to compare the forecast with the actual result for this held-back season."}</p>
            </section>

            <section className="panel models-panel" id="models">
              <div className="panel-header"><h2>Model Comparison</h2><span className="honesty-chip">No random split</span></div>
              <div className="model-cards">
                {topMetrics.map((metric) => (
                  <article key={metric.id} className={metric.id === "rank_calibrated_lasso" ? "selected" : ""}>
                    <div><ModelIcon id={metric.id} /><b>{metric.name}</b>{metric.id === "rank_calibrated_lasso" ? <span>SELECTED</span> : null}</div>
                    <dl><dt>MAE</dt><dd>{metric.maeWins.toFixed(2)} <small>wins</small></dd><dt>RMSE</dt><dd>{metric.rmseWins.toFixed(2)} <small>wins</small></dd></dl>
                  </article>
                ))}
              </div>
              <div className="model-audit-strip" aria-label="Generalization and compression diagnostics">
                <div><small>Current forecast span</small><b>{data.metadata.currentPredictionRangeWins.toFixed(1)} wins</b><em>{data.metadata.previousPredictionRangeWins ? `was ${data.metadata.previousPredictionRangeWins.toFixed(1)}` : "30 teams"}</em></div>
                <div><small>Validation spread captured</small><b>{(data.metadata.validationSpreadRatio * 100).toFixed(1)}%</b><em>{(data.metadata.minimumValidationSpreadRatio * 100).toFixed(0)}% minimum</em></div>
                <div><small>Recent held-out accuracy</small><b>{data.metadata.heldoutMaeWins.toFixed(2)} MAE</b><em>{data.metadata.heldoutRmseWins.toFixed(2)} RMSE</em></div>
                <div><small>Market benchmark</small><b>{data.backtest.overall.marketMaeWins.toFixed(2)} MAE</b><em>still stronger overall</em></div>
              </div>
              <p className="model-disclaimer">Lasso retained {data.metadata.selectedFeatureCount} of 10 signals. The {(data.metadata.rankCalibrationBlend * 100).toFixed(0)}% rank-distribution blend was selected chronologically after rejecting compressed candidates below the spread floor. Extreme-team bias remains: under-30 teams {data.metadata.heldoutTailBias.under30BiasWins >= 0 ? "+" : ""}{data.metadata.heldoutTailBias.under30BiasWins.toFixed(1)} wins and 50-win teams {data.metadata.heldoutTailBias.fiftyPlusBiasWins >= 0 ? "+" : ""}{data.metadata.heldoutTailBias.fiftyPlusBiasWins.toFixed(1)} wins on the recent holdout.</p>
            </section>

            <section className="panel backtest-panel" id="backtest">
              <div className="panel-header chart-header"><div><h2>Backtest Performance</h2><span>$10 on every model side, −110 assumed · shortened seasons schedule-adjusted</span></div><div className="legend"><span className="model-key" /> Model MAE <span className="market-key" /> Market MAE</div></div>
              <BacktestChart years={data.backtest.years} />
              <div className="backtest-stats">
                <div><small>Settled record</small><b>{data.backtest.overall.wins}–{data.backtest.overall.losses}</b></div>
                <div><small>Win rate</small><b>{(data.backtest.overall.winRate * 100).toFixed(1)}%</b></div>
                <div><small>Net on $10 bets</small><b className={data.backtest.overall.netProfit >= 0 ? "positive" : "negative"}>{money(data.backtest.overall.netProfit)}</b></div>
                <div><small>ROI</small><b>{(data.backtest.overall.roi * 100).toFixed(1)}%</b></div>
              </div>
            </section>
          </div>

          {swingInfoOpen && typeof document !== "undefined" ? createPortal((
            <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.currentTarget === event.target) setSwingInfoOpen(false); }}>
              <section className="swing-info-dialog" id="swing-info-dialog" role="dialog" aria-modal="true" aria-labelledby="swing-info-title" aria-describedby="swing-info-description">
                <button ref={swingCloseRef} className="dialog-close" type="button" aria-label="Close 3-point swing explanation" onClick={() => setSwingInfoOpen(false)}>×</button>
                <div className="dialog-icon"><Icon name="info" size={22} /></div>
                <div>
                  <span className="dialog-eyebrow">MODEL SENSITIVITY</span>
                  <h2 id="swing-info-title">What does 3-PT Swing mean?</h2>
                  <p id="swing-info-description">It estimates how much the team&apos;s projected win total changes when this player&apos;s <strong>2K overall rating moves by three points</strong>, with the other roster inputs held constant.</p>
                  <div className="swing-example"><span>Example</span><b>+0.7</b><p>A +3 OVR change adds about 0.7 projected wins; a −3 OVR change removes about 0.7.</p></div>
                  <p className="dialog-note">This is a local Lasso-model scenario—not the player&apos;s three-point shooting, and not a claim that the player personally creates that many wins.</p>
                </div>
              </section>
            </div>
          ), document.body) : null}

          <footer className="data-footer">
            <div><Icon name="verified" size={18} /><span><b>{isCurrentSeason ? "Roster audit passed" : "Historical matrix loaded"}</b> · {isCurrentSeason ? `${data.audit.officialMoveChecks.length} official 2026 moves checked · ${data.audit.verifiedHeadshotCount}/${data.audit.playerCount} headshots explicitly published by ESPN` : `${selectedSeason} top-10 ratings, positions, prediction, and actual record`}</span></div>
            <nav aria-label="Data sources">{data.metadata.sources.map((source) => <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.label}<Icon name="external" size={13} /></a>)}</nav>
          </footer>
        </div>
      </section>
    </main>
  );
}
