/**
 * Pure path helpers shared by the citation resolver and the path index.
 *
 * No `vscode` import, so these are unit-tested in plain Node and reusable by the
 * planned JetBrains sidecar.
 */

/** Normalize a workspace-relative path: forward slashes, no leading `./` or `/`, no trailing `/`. */
export function normalizeRelativePath(path: string): string {
  return path
    .trim()
    .replace(/\\/g, "/")
    .replace(/^\.\//, "")
    .replace(/^\/+/, "")
    .replace(/\/+$/, "");
}

/** The final path segment (basename) of a path or document name. */
export function basenameOf(path: string): string {
  const segments = path.split(/[\\/]/);
  return (segments[segments.length - 1] ?? path).trim();
}

/** A basename with its final extension removed (e.g. `composer.json` → `composer`). */
export function stemOf(name: string): string {
  return basenameOf(name).replace(/\.[^.]+$/, "");
}

/**
 * Indices of `candidateRelativePaths` that also appear in `knownPaths` (both
 * normalized), preserving candidate order. Used to prefer files the user
 * actually ingested when several share a basename.
 */
export function knownPathMatches(candidateRelativePaths: string[], knownPaths: string[]): number[] {
  const known = new Set(knownPaths.map(normalizeRelativePath).filter((path) => path.length > 0));
  const matches: number[] = [];
  candidateRelativePaths.forEach((path, index) => {
    if (known.has(normalizeRelativePath(path))) {
      matches.push(index);
    }
  });
  return matches;
}
