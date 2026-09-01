import nextJest from "next/jest.js";

// next/jest wires up SWC and the CSS/asset stubs, so tests compile through the
// same pipeline the app does. It does not carry the tsconfig paths across; see
// moduleNameMapper below.
const createJestConfig = nextJest({ dir: "./" });

/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jest-environment-jsdom",
  // `output: "standalone"` emits .next/standalone/package.json with the same
  // name as the root one, which trips a haste collision warning on every run
  // after a build.
  modulePathIgnorePatterns: ["<rootDir>/.next/"],
  // next/jest does not carry the tsconfig "paths" aliases across, so map
  // them here. Without this, every "@/..." import fails to resolve.
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
};

export default createJestConfig(config);
