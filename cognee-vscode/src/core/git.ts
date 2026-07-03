/**
 * Pure helpers for reading a repository's identity from its git config text.
 *
 * The filesystem lookup (locating `.git/config`, handling worktrees) lives in
 * the editor layer; this parsing is kept here so it is testable and reusable.
 */

/** Extract the `[remote "origin"] url = ...` value from a git config file's contents. */
export function parseOriginUrl(gitConfig: string): string | null {
  let inOrigin = false;
  for (const line of gitConfig.split(/\r?\n/)) {
    const section = /^\s*\[(.+?)\]\s*$/.exec(line);
    if (section) {
      inOrigin = /^remote\s+"origin"$/.test(section[1].trim());
      continue;
    }
    if (inOrigin) {
      const url = /^\s*url\s*=\s*(.+?)\s*$/.exec(line);
      if (url) {
        return url[1].trim();
      }
    }
  }
  return null;
}
