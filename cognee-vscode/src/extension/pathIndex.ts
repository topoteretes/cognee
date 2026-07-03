import { basenameOf, normalizeRelativePath, stemOf } from "./paths";

/**
 * Minimal persistence contract. Structurally satisfied by `vscode.Memento`
 * (workspace state), and by a plain object in tests — so this module needs no
 * `vscode` import and is fully unit-testable.
 */
export interface KeyValueStore {
  get<T>(key: string): T | undefined;
  update(key: string, value: unknown): Thenable<void>;
}

type IndexShape = Record<string, string[]>;

/**
 * Maps a document basename to the workspace-relative path(s) that were ingested
 * for it, persisted per workspace.
 *
 * Cognee's recall citations carry only a document *basename* (its metadata
 * reduces the source to a display name and does not surface the directory
 * path). This index lets a citation resolve directly to the exact file the user
 * actually remembered — no snippet guessing, no picker — even when the workspace
 * has several files of that name.
 */
export class PathIndex {
  private static readonly STORAGE_KEY = "cognee.pathIndex";

  constructor(private readonly store: KeyValueStore) {}

  /** Record that a workspace-relative path was ingested. Idempotent. */
  async record(relativePath: string): Promise<void> {
    const normalized = normalizeRelativePath(relativePath);
    if (!normalized) {
      return;
    }
    const key = basenameOf(normalized);
    const index = this.read();
    const paths = new Set(index[key] ?? []);
    paths.add(normalized);
    index[key] = [...paths].sort();
    await this.store.update(PathIndex.STORAGE_KEY, index);
  }

  /**
   * Workspace-relative paths previously ingested for a cited document name.
   * Falls back to matching on the extension-less stem, in case the backend
   * dropped the extension from the document name.
   */
  pathsFor(documentName: string): string[] {
    const index = this.read();
    const key = basenameOf(documentName);

    const direct = index[key];
    if (direct && direct.length > 0) {
      return direct;
    }

    const stem = stemOf(key);
    if (!stem) {
      return [];
    }
    const matches = new Set<string>();
    for (const [indexedName, paths] of Object.entries(index)) {
      if (stemOf(indexedName) === stem) {
        for (const path of paths) {
          matches.add(path);
        }
      }
    }
    return [...matches].sort();
  }

  private read(): IndexShape {
    const stored = this.store.get<IndexShape>(PathIndex.STORAGE_KEY);
    // Return a shallow clone so callers never mutate the persisted object in place.
    return stored ? { ...stored } : {};
  }
}
