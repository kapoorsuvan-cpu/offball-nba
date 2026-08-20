"use client";

import Image from "next/image";
import { useState } from "react";
import styles from "./scout.module.css";

type PlayerImageProps = {
  name: string;
  src: string | null;
  fallbackSrc?: string | null;
  size?: number;
  priority?: boolean;
};

export function PlayerImage({ name, src, fallbackSrc, size = 42, priority = false }: PlayerImageProps) {
  const [failedSources, setFailedSources] = useState<string[]>([]);
  const activeSource = [src, fallbackSrc].find(
    (candidate): candidate is string => Boolean(candidate && !failedSources.includes(candidate)),
  );
  const initials = name
    .split(" ")
    .slice(0, 2)
    .map((part) => part[0])
    .join("");

  return (
    <span className={styles.playerImage} style={{ width: size, height: size }}>
      {!activeSource ? (
        <span className={styles.playerInitials}>{initials}</span>
      ) : (
        <Image
          src={activeSource}
          alt={`${name} headshot`}
          width={size}
          height={size}
          sizes={`${size}px`}
          preload={priority}
          onError={() => setFailedSources((current) => [...new Set([...current, activeSource])])}
        />
      )}
    </span>
  );
}

export function TeamLogo({ name, src, size = 34 }: { name: string; src: string; size?: number }) {
  return (
    <span className={styles.teamLogo} style={{ width: size, height: size }}>
      <Image src={src} alt={`${name} logo`} width={size} height={size} sizes={`${size}px`} />
    </span>
  );
}
