#!/usr/bin/env node
/**
 * Regression guard for the recurring "cognee-frontend fails to build at
 * release" root cause (issues #1869, #2413, #2442, #2605, #2709, #2832).
 *
 * This is a fast, dependency-free check (Node built-ins only) that asserts
 * the deterministic, file-level invariants that have broken the frontend
 * build in the past. It does NOT replace `next build` (run in CI via
 * .github/workflows/frontend_build.yml) but catches the specific
 * regressions cheaply and with clear error messages.
 *
 * Checks:
 *   1. (#2832) package-lock.json root dependencies/devDependencies match
 *      package.json exactly, so `npm ci` will not fail.
 *   2. (#2413/#2442) every bare (non-relative, non-alias) package imported
 *      in src/ is resolvable: declared in package.json deps/devDeps OR
 *      present as a node (transitive) in package-lock.json.
 *   3. (#2605/#2709) every relative and "@/" alias import in src/ resolves
 *      to a real file using the EXACT on-disk casing (case-sensitive),
 *      matching Linux CI / Turbopack behaviour.
 *   4. (#1869) src/middleware.ts does not use eval-based / Auth0 edge
 *      middleware that is blocked under Docker CSP.
 *
 * Exit code 0 = all invariants hold, non-zero = a regression was found.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const SRC = path.join(ROOT, "src");

const EXTS = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json", ".css", ".scss"];
const CODE_EXTS = [".ts", ".tsx", ".js", ".jsx"];

const failures = [];
const fail = (msg) => failures.push(msg);

function readJSON(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, out);
    else if (CODE_EXTS.includes(path.extname(entry.name))) out.push(full);
  }
  return out;
}

/** Collect import/require/dynamic-import specifiers, ignoring comments. */
function extractSpecifiers(text) {
  // Strip block and line comments so commented-out imports are ignored.
  const stripped = text
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1");
  const specs = [];
  // Statement-level `import ... from "x"` / `export ... from "x"` anchored at
  // the start of a line (after optional whitespace) so that `from` appearing
  // inside object literals or strings is not matched. The clause between the
  // keyword and `from` is restricted to identifiers/braces/commas/whitespace
  // and the `as` alias keyword — never a quote — so embedded string literals
  // cannot leak in.
  const fromRe = /^[ \t]*(?:import|export)\b[\sA-Za-z0-9_${},*]*?\bfrom\s*['"]([^'"]+)['"]/gm;
  // Bare side-effect import: `import "x";`
  const sideEffectRe = /^[ \t]*import\s+['"]([^'"]+)['"]/gm;
  // Dynamic import and require with a string literal argument.
  const dynRe = /\b(?:import|require)\s*\(\s*['"]([^'"]+)['"]\s*\)/g;
  for (const re of [fromRe, sideEffectRe, dynRe]) {
    let m;
    while ((m = re.exec(stripped)) !== null) specs.push(m[1]);
  }
  return specs;
}

/** Verify a resolved path matches actual on-disk casing component by component. */
function existsExactCase(absPath) {
  let cur = path.parse(absPath).root || ".";
  const rel = path.relative(cur, absPath);
  for (const part of rel.split(path.sep)) {
    if (!part) continue;
    let entries;
    try {
      entries = fs.readdirSync(cur);
    } catch {
      return false;
    }
    if (!entries.includes(part)) return false;
    cur = path.join(cur, part);
  }
  return true;
}

/** Resolve a module-style target (no ext) to a concrete file path, or null. */
function resolveTarget(target) {
  if (fs.existsSync(target) && fs.statSync(target).isFile()) return target;
  for (const e of EXTS) {
    if (fs.existsSync(target + e)) return target + e;
  }
  if (fs.existsSync(target) && fs.statSync(target).isDirectory()) {
    for (const e of EXTS) {
      const idx = path.join(target, "index" + e);
      if (fs.existsSync(idx)) return idx;
    }
  }
  return null;
}

// ── Check 1: lockfile in sync (#2832) ──────────────────────────────────
const pkg = readJSON(path.join(ROOT, "package.json"));
const lockPath = path.join(ROOT, "package-lock.json");
let lockNodeNames = new Set();
if (!fs.existsSync(lockPath)) {
  fail("#2832: package-lock.json is missing — `npm ci` will fail.");
} else {
  const lock = readJSON(lockPath);
  const rootNode = (lock.packages && lock.packages[""]) || {};
  const lockDeps = { ...(rootNode.dependencies || {}), ...(rootNode.devDependencies || {}) };
  const pkgDeps = { ...(pkg.dependencies || {}), ...(pkg.devDependencies || {}) };
  for (const [name, spec] of Object.entries(pkgDeps)) {
    if (!(name in lockDeps)) {
      fail(`#2832: "${name}" in package.json is missing from package-lock.json root — run \`npm install\` to resync.`);
    } else if (lockDeps[name] !== spec) {
      fail(`#2832: version spec for "${name}" differs (package.json="${spec}", lock="${lockDeps[name]}") — resync lockfile.`);
    }
  }
  for (const name of Object.keys(lockDeps)) {
    if (!(name in pkgDeps)) {
      fail(`#2832: "${name}" in package-lock.json root is not in package.json — resync lockfile.`);
    }
  }
  // Collect all installed package names (incl. transitive) for import resolution.
  for (const key of Object.keys(lock.packages || {})) {
    const idx = key.lastIndexOf("node_modules/");
    if (idx !== -1) lockNodeNames.add(key.slice(idx + "node_modules/".length));
  }
}

const declaredPkgs = new Set([
  ...Object.keys(pkg.dependencies || {}),
  ...Object.keys(pkg.devDependencies || {}),
]);

const NODE_BUILTINS = new Set([
  "fs", "path", "url", "os", "crypto", "util", "stream", "http", "https",
  "events", "buffer", "process", "child_process", "zlib", "net", "tls",
]);

function bareToPackageName(spec) {
  if (spec.startsWith("@")) return spec.split("/").slice(0, 2).join("/");
  return spec.split("/")[0];
}

// ── Walk source and run checks 2 & 3 ───────────────────────────────────
const files = walk(SRC);
for (const file of files) {
  const text = fs.readFileSync(file, "utf8");
  for (const spec of extractSpecifiers(text)) {
    let target;
    if (spec.startsWith("@/")) {
      target = path.join(SRC, spec.slice(2));
    } else if (spec.startsWith(".")) {
      target = path.normalize(path.join(path.dirname(file), spec));
    } else {
      // Check 2: bare package import is resolvable (#2413/#2442).
      const name = bareToPackageName(spec);
      if (name.startsWith("node:") || NODE_BUILTINS.has(name)) continue;
      if (!declaredPkgs.has(name) && !lockNodeNames.has(name)) {
        fail(`#2413/#2442: ${path.relative(ROOT, file)} imports "${spec}" but package "${name}" is not declared in package.json nor present in package-lock.json.`);
      }
      continue;
    }
    // Check 3: relative/alias import resolves with exact casing (#2605/#2709).
    const resolved = resolveTarget(target);
    if (!resolved) {
      fail(`#2442: ${path.relative(ROOT, file)} imports "${spec}" which does not resolve to any file under src/.`);
    } else if (!existsExactCase(resolved)) {
      fail(`#2605/#2709: ${path.relative(ROOT, file)} imports "${spec}" with casing that does not match the on-disk path "${path.relative(ROOT, resolved)}". This breaks on case-sensitive filesystems (Linux CI / Turbopack).`);
    }
  }
}

// ── Check 4: no eval-based / Auth0 edge middleware (#1869) ─────────────
const middlewarePath = path.join(SRC, "middleware.ts");
if (fs.existsSync(middlewarePath)) {
  const mw = fs.readFileSync(middlewarePath, "utf8").replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  if (/\beval\s*\(/.test(mw)) {
    fail("#1869: src/middleware.ts uses eval(), which is blocked under the Docker CSP and breaks edge middleware.");
  }
  if (/auth0|withMiddlewareAuthRequired/i.test(mw)) {
    fail("#1869: src/middleware.ts references Auth0 edge middleware, which previously broke the Docker build under CSP. Local mode must use a pass-through middleware.");
  }
}

// ── Report ─────────────────────────────────────────────────────────────
if (failures.length > 0) {
  console.error(`\nFrontend build invariant check FAILED (${failures.length} issue(s)):\n`);
  for (const f of failures) console.error("  - " + f);
  console.error("");
  process.exit(1);
}
console.log(`Frontend build invariants OK: scanned ${files.length} source files.`);
