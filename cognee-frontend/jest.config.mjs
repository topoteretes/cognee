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
  // Mocks carry their call counts between tests otherwise, so a suite that
  // asserts on call counts (useBrainGraph.test.tsx) passes or fails depending
  // on what ran before it.
  clearMocks: true,
};

// d3 and its transitive deps ship ESM only ("type": "module"), so any suite
// that reaches them (modules/business/* imports d3-color) dies on `export`
// with an "unexpected token" before a single assertion runs.
//
// next/jest ignores all of node_modules and only *appends* whatever
// transformIgnorePatterns the config passes in ("Custom config can append to
// transformIgnorePatterns but not modify it" — next/dist/build/jest). The list
// is an OR, so its own entry always wins and appending achieves nothing. Next's
// escape hatch is `transpilePackages` in next.config.ts, but that changes how
// the production bundle is built to fix a test-only problem. So widen the
// resolved patterns here instead, after next/jest has produced them.
//
// Written to extend whatever next/jest generated rather than to replace it:
// the exact shape depends on the transpilePackages it derives (today "geist"),
// and hardcoding a replacement would silently drop those the next time they
// change. The CSS-module entry it also sets is left alone.
const ESM_DEPS = "d3-.*|internmap|delaunator|robust-predicates";

function allowEsmDeps(pattern) {
  // "/node_modules/(?!.pnpm)(?!(geist)/)" and its ".pnpm/(?!(geist)@)" sibling
  if (pattern.includes("/node_modules/") && pattern.includes("(?!(")) {
    return pattern.replace(/\(\?!\(([^)]*)\)/g, (_match, alternatives) => `(?!(${alternatives}|${ESM_DEPS})`);
  }
  // The shape used when no packages are transpiled at all
  if (pattern === "/node_modules/") {
    return `/node_modules/(?!(${ESM_DEPS})/)`;
  }
  return pattern;
}

export default async function jestConfig() {
  const resolved = await createJestConfig(config)();
  return {
    ...resolved,
    transformIgnorePatterns: (resolved.transformIgnorePatterns ?? []).map(allowEsmDeps),
  };
}
