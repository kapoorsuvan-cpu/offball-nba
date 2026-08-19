"use client";

import Image from "next/image";
import { useState } from "react";
import styles from "./scout.module.css";

type PlayerImageProps = {
  name: string;
  src: string | null;
  size?: number;
  priority?: boolean;
};

export function PlayerImage({ name, src, size = 42, priority = false }: PlayerImageProps) {
  const [failed, setFailed] = useState(false);
  const initials = name
    .split(" ")
    .slice(0, 2)
    .map((part) => part[0])
    .join("");

  return (
    <span className={styles.playerImage} style={{ width: size, height: size }}>
      {!src || failed ? (
        <span className={styles.playerInitials}>{initials}</span>
      ) : (
        <Image
          src={src}
          alt={`${name} headshot`}
          width={size}
          height={size}
          sizes={`${size}px`}
          preload={priority}
          onError={() => setFailed(true)}
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
