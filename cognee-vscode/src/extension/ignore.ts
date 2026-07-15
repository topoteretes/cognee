import ignore from "ignore";

/**
 * Build a predicate that reports whether a workspace-relative path is excluded
 * by the given ignore-file contents. Inputs use `.gitignore` / `.cogneeignore`
 * syntax — globs, `!` negation, `/` anchoring, and trailing `/` for directories —
 * matched by the `ignore` library so the semantics are exactly git's.
 *
 * Returns undefined when none of the inputs contain any rules, so the caller can
 * skip filtering entirely. Kept `vscode`-free so it is unit-tested in plain Node
 * and reusable by the planned JetBrains sidecar.
 */
export function buildIgnoreFilter(
  ...contents: (string | undefined)[]
): ((relativePath: string) => boolean) | undefined {
  const matcher = ignore();
  let hasRules = false;
  for (const content of contents) {
    if (content && content.trim()) {
      matcher.add(content);
      hasRules = true;
    }
  }
  if (!hasRules) {
    return undefined;
  }
  return (relativePath) => {
    // `ignore` expects a relative, forward-slashed path and throws on empty or
    // absolute input; a blank path is never considered ignored.
    const normalized = relativePath.replace(/\\/g, "/").replace(/^\/+/, "");
    return normalized.length > 0 && matcher.ignores(normalized);
  };
}
