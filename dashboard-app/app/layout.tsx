import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://offball-nba.vercel.app"),
  title: {
    default: "OFFBALL — NBA Win Predictor",
    template: "%s | OFFBALL",
  },
  description:
    "Explore 2026-27 NBA rosters, current 2K27 ratings, ESPN headshots, and spread-calibrated win projections.",
  openGraph: {
    title: "OFFBALL — NBA Win Predictor",
    description:
      "Current NBA rosters, 2K27 ratings, anti-compression model diagnostics, and honest historical backtests.",
    images: [{ url: "/og-v2.png", width: 1200, height: 630 }],
  },
  twitter: {
    card: "summary_large_image",
    images: ["/og-v2.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
