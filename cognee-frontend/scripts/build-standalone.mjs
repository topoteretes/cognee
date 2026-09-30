// Bundles src/standalone/business-standalone.tsx into the Python package so
// the backend's `visualize` endpoint and `cognee.visualize_graph()` can serve
// the Business canvas as a self-contained page. React and d3 are bundled in;
// the output is committed, so run this whenever src/modules/business changes.
import { build } from "esbuild";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outfile = path.resolve(
  frontendRoot,
  "../cognee/modules/visualization/views/business_standalone.js",
);

await build({
  entryPoints: [path.join(frontendRoot, "src/standalone/business-standalone.tsx")],
  outfile,
  bundle: true,
  minify: true,
  format: "iife",
  target: "es2020",
  jsx: "automatic",
  alias: { "@": path.join(frontendRoot, "src") },
  define: { "process.env.NODE_ENV": '"production"' },
  legalComments: "none",
  logLevel: "info",
});
