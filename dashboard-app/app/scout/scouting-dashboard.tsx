"use client";

import Image from "next/image";
import { useMemo, useState } from "react";
import { CourtHeatmap } from "./court-heatmap";
import { PlayerImage, TeamLogo } from "./player-image";
import styles from "./scout.module.css";
import type { ScoutPlayer, ScoutTeam, ScoutingData, ZoneStat } from "./types";

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatPct(value: number | null) {
  return value === null ? "—" : `${value.toFixed(1)}%`;
}

function signed(value: number | null) {
  if (value === null) return "—";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

function unitPlayers(team: ScoutTeam, ids: string[]) {
  const byId = new Map(team.players.map((player) => [player.id, player]));
  return ids.map((id) => byId.get(id)).filter((player): player is ScoutPlayer => Boolean(player));
}

function qualifyingExtreme(zones: ZoneStat[], direction: "best" | "worst") {
  const qualifying = zones.filter(
    (zone): zone is ZoneStat & { delta: number } => zone.attempts >= 20 && zone.delta !== null,
  );
  return qualifying.toSorted((a, b) =>
    direction === "best" ? b.delta - a.delta : a.delta - b.delta,
  )[0];
}

function Metric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className={styles.metric}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function UnitCard({ title, players, tone }: { title: string; players: ScoutPlayer[]; tone: "green" | "blue" }) {
  return (
    <section className={`${styles.unitCard} ${tone === "green" ? styles.unitGreen : styles.unitBlue}`}>
      <div className={styles.unitHeading}>
        <div>
          <p>Projected</p>
          <h3>{title}</h3>
        </div>
        <span>{players.length} players</span>
      </div>
      <div className={styles.unitPlayers}>
        {players.map((player) => (
          <div className={styles.unitPlayer} key={player.id}>
            <PlayerImage name={player.name} src={player.headshotUrl} size={54} />
            <strong>{player.name}</strong>
            <span>{player.position} · {player.rating} OVR</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function ScoutingDashboard({ data }: { data: ScoutingData }) {
  const initialTeam = data.teams.find((team) => team.abbreviation === "TOR") ?? data.teams[0];
  const [selectedTeamAbbreviation, setSelectedTeamAbbreviation] = useState(initialTeam.abbreviation);
  const [selectedPlayerId, setSelectedPlayerId] = useState(
    initialTeam.projected.starters[0] ?? initialTeam.players[0].id,
  );
  const [teamQuery, setTeamQuery] = useState("");

  const team = useMemo(
    () => data.teams.find((item) => item.abbreviation === selectedTeamAbbreviation) ?? data.teams[0],
    [data.teams, selectedTeamAbbreviation],
  );
  const rotationIds = useMemo(
    () => [...team.projected.starters, ...team.projected.secondUnit],
    [team],
  );
  const rotation = useMemo(() => unitPlayers(team, rotationIds), [rotationIds, team]);
  const rosterPlayers = useMemo(() => {
    const rotationIdSet = new Set(rotationIds);
    return [...rotation, ...team.players.filter((item) => !rotationIdSet.has(item.id))];
  }, [rotation, rotationIds, team.players]);
  const player = team.players.find((item) => item.id === selectedPlayerId) ?? rotation[0] ?? team.players[0];
  const starters = unitPlayers(team, team.projected.starters);
  const secondUnit = unitPlayers(team, team.projected.secondUnit);

  const visibleTeams = useMemo(() => {
    const query = teamQuery.trim().toLowerCase();
    if (!query) return data.teams;
    return data.teams.filter(
      (item) =>
        item.name.toLowerCase().includes(query) || item.abbreviation.toLowerCase().includes(query),
    );
  }, [data.teams, teamQuery]);

  const strongestZone = qualifyingExtreme(player.offense.zones, "best");
  const weakestZone = qualifyingExtreme(player.offense.zones, "worst");
  const defenseTarget = qualifyingExtreme(team.defenseZones, "best");
  const unavailable = team.players.filter((item) => item.status !== "Active");

  const chooseTeam = (nextTeam: ScoutTeam) => {
    setSelectedTeamAbbreviation(nextTeam.abbreviation);
    setSelectedPlayerId(nextTeam.projected.starters[0] ?? nextTeam.players[0].id);
  };

  return (
    <main className={styles.shell} style={{ "--team-color": team.color } as React.CSSProperties}>
      <aside className={styles.sidebar}>
        <div className={styles.brand}>
          <Image
            className={styles.brandLogo}
            src="/offball-logo.png"
            alt="OFFBALL"
            width={824}
            height={238}
            priority
          />
        </div>

        <div className={styles.teamPickerHeading}>
          <div>
            <p>League</p>
            <h2>All 30 teams</h2>
          </div>
        </div>
        <label className={styles.searchBox}>
          <span className={styles.srOnly}>Search teams</span>
          <input
            value={teamQuery}
            onChange={(event) => setTeamQuery(event.target.value)}
            placeholder="Search team…"
          />
        </label>

        <div className={styles.teamList}>
          {visibleTeams.map((item) => (
            <button
              className={item.abbreviation === team.abbreviation ? styles.activeTeam : undefined}
              key={item.id}
              onClick={() => chooseTeam(item)}
              aria-pressed={item.abbreviation === team.abbreviation}
            >
              <TeamLogo name={item.name} src={item.logoUrl} size={28} />
              <span>
                <strong>{item.abbreviation}</strong>
                <small>{item.shortName}</small>
              </span>
            </button>
          ))}
        </div>
      </aside>

      <section className={styles.workspace}>
        <header className={styles.topbar}>
          <div className={styles.teamIdentity}>
            <TeamLogo name={team.name} src={team.logoUrl} size={54} />
            <div>
              <p>{data.metadata.rosterSeason} roster</p>
              <h1>{team.name} <span>daily scout</span></h1>
            </div>
          </div>
          <div className={styles.sourceStatus}>
            <span className={styles.statusDot} />
            <div>
              <strong>Data snapshot ready</strong>
              <small>Built {formatDate(data.metadata.generatedAt)}</small>
            </div>
          </div>
          <div className={styles.seasonBadge}>
            <span>Shooting sample</span>
            <strong>{data.metadata.statsSeason}</strong>
          </div>
        </header>

        <div className={styles.dashboardGrid}>
          <section className={styles.rotationColumn}>
            <div className={styles.panelHeading}>
              <div>
                <p>{team.players.length} players · projected depth first</p>
                <h2>Full team roster</h2>
              </div>
              <span>OVR</span>
            </div>
            <div className={styles.rotationList}>
              {rosterPlayers.map((rotationPlayer, index) => (
                <button
                  key={rotationPlayer.id}
                  onClick={() => setSelectedPlayerId(rotationPlayer.id)}
                  className={rotationPlayer.id === player.id ? styles.activePlayer : undefined}
                  aria-pressed={rotationPlayer.id === player.id}
                >
                  <span className={styles.rotationRank}>{index + 1}</span>
                  <PlayerImage name={rotationPlayer.name} src={rotationPlayer.headshotUrl} size={42} />
                  <span className={styles.rotationName}>
                    <strong>{rotationPlayer.name}</strong>
                    <small>
                      {index < 5
                        ? "Starter"
                        : index < 10
                          ? "Second unit"
                          : rotationPlayer.status === "Active"
                            ? "Roster"
                            : rotationPlayer.status} · {rotationPlayer.position} · {rotationPlayer.offense.attempts} FGA
                    </small>
                  </span>
                  <strong className={styles.rating}>{rotationPlayer.rating}</strong>
                </button>
              ))}
            </div>

            <section className={styles.injuryCard}>
              <div className={styles.injuryHeading}>
                <div>
                  <p>Roster status</p>
                  <h3>Injury feed</h3>
                </div>
                <span>Planned</span>
              </div>
              {unavailable.length ? (
                unavailable.slice(0, 4).map((item) => (
                  <div className={styles.statusRow} key={item.id}>
                    <span className={styles.statusAlert} />
                    <strong>{item.name}</strong>
                    <small>{item.status}</small>
                  </div>
                ))
              ) : (
                <p className={styles.emptyStatus}>No unavailable players in the roster snapshot.</p>
              )}
              <small className={styles.feedNote}>Live day-by-day injury reporting is not connected yet.</small>
            </section>
          </section>

          <section className={styles.analysisColumn}>
            <div className={styles.playerHeader}>
              <PlayerImage name={player.name} src={player.headshotUrl} size={76} priority />
              <div>
                <p>Selected player</p>
                <h2>{player.name}</h2>
                <span>#{player.jersey ?? "—"} · {player.positions.join("/")} · {player.rating} OVR</span>
              </div>
              <div className={styles.playerSample}>
                <span>2025–26 sample</span>
                <strong>{player.offense.attempts.toLocaleString()} FGA</strong>
              </div>
            </div>

            <div className={styles.heatmapGrid}>
              <CourtHeatmap
                zones={player.offense.zones}
                title={`${player.name} shot profile`}
                eyebrow="Player offense · FG%"
              />
              <CourtHeatmap
                zones={team.defenseZones}
                inverse
                title={`${team.shortName} concessions`}
                eyebrow="Team defense · Opponent FG%"
              />
            </div>

            <div className={styles.dataCues}>
              <div className={styles.cardHeading}>
                <div>
                  <p>Film-room starting points</p>
                  <h3>Data cues</h3>
                </div>
                <span>Min. 20 FGA</span>
              </div>
              <div className={styles.cueGrid}>
                <article>
                  <span className={styles.cueNumber}>01</span>
                  <div>
                    <strong>Take away {strongestZone?.label ?? "best zone"}</strong>
                    <p>
                      {strongestZone
                        ? `${formatPct(strongestZone.pct)} on ${strongestZone.attempts} attempts (${signed(strongestZone.delta)} points vs. league).`
                        : "No qualifying offensive sample for this player."}
                    </p>
                  </div>
                </article>
                <article>
                  <span className={styles.cueNumber}>02</span>
                  <div>
                    <strong>Shade toward {weakestZone?.label ?? "lower-efficiency space"}</strong>
                    <p>
                      {weakestZone
                        ? `${formatPct(weakestZone.pct)} on ${weakestZone.attempts} attempts (${signed(weakestZone.delta)} points vs. league).`
                        : "No qualifying offensive sample for this player."}
                    </p>
                  </div>
                </article>
                <article>
                  <span className={styles.cueNumber}>03</span>
                  <div>
                    <strong>Team target: {defenseTarget?.label ?? "review full film"}</strong>
                    <p>
                      {defenseTarget
                        ? `${team.shortName} allowed ${formatPct(defenseTarget.pct)} there (${signed(defenseTarget.delta)} points vs. league).`
                        : "No qualifying team defensive sample."}
                    </p>
                  </div>
                </article>
              </div>
            </div>
          </section>

          <aside className={styles.detailColumn}>
            <section className={styles.splitCard}>
              <div className={styles.cardHeading}>
                <div>
                  <p>Volume + accuracy</p>
                  <h3>Shot-zone splits</h3>
                </div>
                <span>FREQ · FG%</span>
              </div>
              <div className={styles.splitRows}>
                {player.offense.zones.map((zone) => (
                  <div className={styles.splitRow} key={zone.key}>
                    <span>{zone.label}</span>
                    <div className={styles.frequencyBar}>
                      <i style={{ width: `${Math.min(zone.frequency, 100)}%` }} />
                    </div>
                    <small>{zone.frequency.toFixed(1)}%</small>
                    <strong>{formatPct(zone.pct)}</strong>
                  </div>
                ))}
              </div>
            </section>

            <section className={styles.defenseCard}>
              <div className={styles.cardHeading}>
                <div>
                  <p>NBA matchup tracking</p>
                  <h3>On-ball defense</h3>
                </div>
                <span>{player.defense.possessions.toLocaleString()} POSS</span>
              </div>
              {player.defense.fga ? (
                <div className={styles.metricGrid}>
                  <Metric label="Overall" value={formatPct(player.defense.fgPct)} detail={`${player.defense.fgm}/${player.defense.fga} FG`} />
                  <Metric label="Inside arc" value={formatPct(player.defense.twoPct)} detail={`${player.defense.twoPm}/${player.defense.twoPa} 2FG`} />
                  <Metric label="From three" value={formatPct(player.defense.threePct)} detail={`${player.defense.threePm}/${player.defense.threePa} 3FG`} />
                  <Metric label="Events" value={`${player.defense.turnovers + player.defense.blocks}`} detail={`${player.defense.turnovers} TOV · ${player.defense.blocks} BLK`} />
                </div>
              ) : (
                <p className={styles.noSample}>No qualifying matchup-tracking sample.</p>
              )}
              <p className={styles.methodNote}>
                Direct matchup results are player-level. The defensive court map above is team-level because the source does not attach a defender to each shot coordinate.
              </p>
            </section>

            <section className={styles.provenanceCard}>
              <div>
                <span>Roster</span>
                <strong>{formatDate(data.metadata.sourceRosterGeneratedAt)}</strong>
              </div>
              <div>
                <span>Coverage</span>
                <strong>{data.metadata.playersWithOffense}/{data.metadata.playerCount} players</strong>
              </div>
              <p>Headshots and team marks use the ESPN CDN fields already stored in OFFBALL.</p>
            </section>
          </aside>
        </div>

        <div className={styles.unitsGrid}>
          <UnitCard title="Projected starters" players={starters} tone="green" />
          <UnitCard title="Second unit" players={secondUnit} tone="blue" />
        </div>

        <footer className={styles.footer}>
          <span>OFFBALL Scout · actual {data.metadata.statsSeason} regular-season samples</span>
          <div>
            <a href={data.metadata.depthChartSource}>Depth chart source</a>
            <a href={data.metadata.ratingsSource}>Ratings source</a>
            <a href={data.metadata.shotSource}>Shot detail source</a>
            <a href={data.metadata.matchupSource}>Matchup source</a>
          </div>
        </footer>
      </section>
    </main>
  );
}
