import { createHash } from "node:crypto";

/**
 * Per-repository memory scoping.
 *
 * Each workspace maps to a single, stable Cognee dataset so that project memory
 * is isolated per repository. The dataset name is derived deterministically from
 * the git remote URL when available (stable across clones, renames, and machines)
 * and falls back to the workspace path for repos without a remote.
 *
 * Passing an explicit dataset on every call is what keeps two workspaces open in
 * the same editor from colliding — the server's per-client default is never relied on.
 */

export interface ScopeInput {
  /** Absolute path of the workspace root. */
  workspaceRoot: string;
  /** The repository's `origin` remote URL, if any. */
  gitRemote?: string | null;
  /** An explicit dataset name that overrides the derived one. */
  override?: string | null;
  /** Dataset name prefix (default: "vscode"). */
  prefix?: string;
}

const DEFAULT_PREFIX = "vscode";
const HASH_LENGTH = 16;

/**
 * Normalize a git remote so that equivalent URLs (ssh vs https, with/without
 * `.git`, trailing slashes, case) map to the same key.
 *
 * Examples (all normalize to `github.com/user/repo`):
 *   - git@github.com:user/repo.git
 *   - https://github.com/user/repo.git
 *   - ssh://git@github.com/user/repo/
 */
export function normalizeGitRemote(remote: string | null | undefined): string | null {
  if (!remote) {
    return null;
  }
  let value = remote.trim();
  if (!value) {
    return null;
  }

  // Drop a URL scheme (https://, ssh://, git://, ...).
  value = value.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//, "");
  // Drop userinfo (e.g. "git@").
  value = value.replace(/^[^@/]+@/, "");
  // Convert scp-like "host:path" to "host/path" (only when ':' precedes a
  // non-digit, so an explicit ssh port like ":22/" is preserved).
  value = value.replace(/:(?=\D)/, "/");
  // Lowercase, then strip trailing slashes and a trailing ".git".
  value = value
    .toLowerCase()
    .replace(/\/+$/, "")
    .replace(/\.git$/, "")
    .replace(/\/+$/, "");

  return value || null;
}

/** Normalize a filesystem path so separators and trailing slashes don't affect the hash. */
export function normalizeWorkspaceRoot(root: string): string {
  return root.trim().replace(/\\/g, "/").replace(/\/+$/, "");
}

/** Coerce an arbitrary string into a Cognee-safe dataset name. */
export function sanitizeDatasetName(name: string): string {
  const cleaned = name
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/_+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || "workspace";
}

/**
 * Derive the dataset name for a workspace.
 *
 * Precedence: explicit `override` → git remote hash → workspace-path hash.
 */
export function deriveDatasetName(input: ScopeInput): string {
  const prefix = sanitizeDatasetName(input.prefix ?? DEFAULT_PREFIX);

  const override = input.override?.trim();
  if (override) {
    return sanitizeDatasetName(override);
  }

  const key = normalizeGitRemote(input.gitRemote) ?? normalizeWorkspaceRoot(input.workspaceRoot);
  const hash = createHash("sha256").update(key, "utf8").digest("hex").slice(0, HASH_LENGTH);
  return `${prefix}_${hash}`;
}
