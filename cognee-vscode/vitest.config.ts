import { defineConfig } from "vitest/config";

/**
 * Core unit tests run in a plain Node environment against a mocked Cognee
 * backend. The `src/core` layer never imports `vscode`, so no editor host is
 * required — this keeps the suite fast and CI-friendly (no live keys, no LLM).
 */
export default defineConfig({
  test: {
    include: ["src/**/*.test.ts"],
    environment: "node",
    globals: false,
  },
});
