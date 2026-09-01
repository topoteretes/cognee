import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit .next/standalone: a self-contained server bundle with only the
  // node_modules the app actually reaches. This is what the Docker runtime
  // stage copies, and it is why the published image ships no dev dependencies.
  output: "standalone",
  images: {
    remotePatterns: [{
      protocol: "https",
      hostname: "lh3.googleusercontent.com",
    }],
  },
};

export default nextConfig;
