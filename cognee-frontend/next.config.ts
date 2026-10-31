import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit .next/standalone: a self-contained server bundle with only the
  // node_modules the app actually reaches. The public repo's Dockerfile copies
  // exactly that path in its runtime stage, so dropping this line does not
  // degrade the image, it fails the build at the COPY. This override replaces
  // the public next.config.ts wholesale, which means anything that file owns
  // and this one omits is not inherited but removed, which is the CLO-703
  // defect one directory above the tree the deletion guard watches.
  output: "standalone",

  // Local mode is the only mode in the OSS build, and the flags have to be
  // baked in here rather than shipped as a .env.local: the public repo
  // gitignores `.env*.local`, so a synced env file is copied onto the runner
  // and then silently dropped by `git add` (CLO-703). Next inlines these at
  // build time, so a prebuilt `cognee -ui` could not pick them up from the
  // environment afterwards either. Reading process.env first keeps both
  // overridable for anyone building the frontend themselves, including from
  // their own untracked .env.local, which Next loads before this file runs.
  env: {
    NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT:
      process.env.NEXT_PUBLIC_IS_CLOUD_ENVIRONMENT ?? "false",
    NEXT_PUBLIC_LOCAL_API_URL:
      process.env.NEXT_PUBLIC_LOCAL_API_URL ?? "http://localhost:8000",
  },

  images: {
    remotePatterns: [{
      protocol: "https",
      hostname: "lh3.googleusercontent.com",
    }],
  },
};

export default nextConfig;
