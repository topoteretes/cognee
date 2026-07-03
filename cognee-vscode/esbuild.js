"use strict";

const esbuild = require("esbuild");

const production = process.argv.includes("--production");
const watch = process.argv.includes("--watch");

/**
 * Build the extension into a single CommonJS bundle that VS Code can load.
 * `vscode` is provided by the host at runtime and must stay external.
 */
async function main() {
  const context = await esbuild.context({
    entryPoints: ["src/extension/extension.ts"],
    bundle: true,
    format: "cjs",
    platform: "node",
    target: "node18",
    outfile: "dist/extension.js",
    external: ["vscode"],
    sourcemap: !production,
    minify: production,
    logLevel: "info",
  });

  if (watch) {
    await context.watch();
  } else {
    await context.rebuild();
    await context.dispose();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
