import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "a.espncdn.com",
        pathname: "/i/**",
      },
      {
        protocol: "https",
        hostname: "cdn.nba.com",
        pathname: "/headshots/**",
      },
    ],
  },
};

export default nextConfig;
