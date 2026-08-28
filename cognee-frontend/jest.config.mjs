import nextJest from "next/jest.js";

// next/jest wires up SWC, CSS/asset stubs and the tsconfig "paths" aliases, so
// tests compile through the same pipeline the app does.
const createJestConfig = nextJest({ dir: "./" });

/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jest-environment-jsdom",
  // next/jest does not carry the tsconfig "paths" aliases across, so map
  // them here. Without this, every "@/..." import fails to resolve.
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
};

export default createJestConfig(config);
