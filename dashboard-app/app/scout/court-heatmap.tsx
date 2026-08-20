import styles from "./scout.module.css";
import type { ZoneStat } from "./types";

function heatColor(zone: ZoneStat, inverse: boolean) {
  if (zone.pct === null || zone.delta === null || zone.attempts < 10) {
    return "rgba(91, 113, 139, 0.32)";
  }
  const adjustedDelta = inverse ? -zone.delta : zone.delta;
  const strength = Math.min(Math.abs(adjustedDelta) / 12, 1);
  return adjustedDelta >= 0
    ? `rgba(39, 196, 126, ${0.22 + strength * 0.6})`
    : `rgba(244, 89, 82, ${0.22 + strength * 0.6})`;
}

function formatPct(value: number | null) {
  return value === null ? "—" : `${value.toFixed(1)}%`;
}

export function CourtHeatmap({
  zones,
  inverse = false,
  title,
  eyebrow,
}: {
  zones: ZoneStat[];
  inverse?: boolean;
  title: string;
  eyebrow: string;
}) {
  const totalAttempts = zones.reduce((sum, zone) => sum + zone.attempts, 0);

  return (
    <section className={styles.heatmapCard} aria-label={title}>
      <div className={styles.cardHeading}>
        <div>
          <p>{eyebrow}</p>
          <h3>{title}</h3>
        </div>
        <span>{inverse ? "Lower is stronger" : "Vs. league average"}</span>
      </div>

      <div className={styles.court}>
        <div className={styles.courtArc} />
        <div className={styles.paintLines} />
        <div className={styles.rimLine} />
        {zones.map((zone) => (
          <div
            className={`${styles.zone} ${styles[zone.key]}`}
            key={zone.key}
            style={{ backgroundColor: heatColor(zone, inverse) }}
            title={`${zone.label}: ${formatPct(zone.pct)} on ${zone.attempts} attempts`}
          >
            <span>{zone.label}</span>
            <strong>{formatPct(zone.pct)}</strong>
            <small>{zone.attempts} FGA</small>
          </div>
        ))}
        {!totalAttempts ? (
          <div className={styles.noCourtSample}>No qualifying sample</div>
        ) : null}
      </div>

      <div className={styles.legend} aria-label="Heatmap legend">
        <span>Vulnerable</span>
        <i className={styles.legendRed} />
        <i className={styles.legendNeutral} />
        <i className={styles.legendGreen} />
        <span>Strong</span>
      </div>
    </section>
  );
}
