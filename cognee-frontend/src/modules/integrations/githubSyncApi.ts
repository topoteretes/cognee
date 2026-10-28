// Browser-side calls to the SDK's GitHub sync policy routes.
import localFetch from "@/modules/instances/localFetch";
import { HttpError } from "@/services/http/errors";

export type GithubSyncMode = "push" | "scheduled" | "manual";

export interface GithubSchedule {
  intervalMinutes: number;
  weekdays?: number[];
  timezone?: string;
  startTime?: string | null;
  endTime?: string | null;
}

// Fields this card does not edit (per-repository branch and path filters) stay
// on the object and are sent back unchanged.
export interface GithubSyncSettings {
  enabled: boolean;
  mode: GithubSyncMode;
  schedule?: GithubSchedule | null;
  firstSync: "immediate" | "scheduled";
  repositories?: Record<string, GithubRepositorySettings> | null;
}

export interface GithubRepositorySettings {
  branch?: string | null;
  includePaths?: string[] | null;
  excludePaths?: string[];
  includeDocuments?: boolean;
}

export interface GithubRepository {
  id: string;
  fullName: string;
  defaultBranch?: string | null;
}

export interface GithubSyncState {
  settings: GithubSyncSettings;
  nextSyncAt?: string | null;
  lastSyncedAt?: string | null;
  syncStatus?: string | null;
}

const BASE = "/v1/integrations/github/sync";

async function json<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await localFetch(`${BASE}/${path}`, options);
  return response.json();
}

export const githubSyncApi = {
  // Null when no GitHub account is connected (the route answers 404).
  state: async (): Promise<GithubSyncState | null> => {
    try {
      return await json<GithubSyncState>("settings");
    } catch (error) {
      if (error instanceof HttpError && error.status === 404) return null;
      throw error;
    }
  },
  repositories: () => json<GithubRepository[]>("repositories"),
  save: (settings: GithubSyncSettings, syncNow: boolean, acknowledgeRemoval = false) =>
    json<GithubSyncState & { syncRequested: boolean }>("settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings, syncNow, acknowledgeRemoval }),
    }),
  // The generic integrations route mints GitHub's app-install URL and sets the nonce cookie.
  authorize: async (): Promise<string> => {
    const response = await localFetch("/v1/integrations/github/authorize", { method: "POST" });
    const { authorizeUrl } = await response.json();
    const url = new URL(authorizeUrl);
    if (url.origin !== "https://github.com") throw new Error("The server returned an unexpected GitHub install URL.");
    return url.href;
  },
  runDue: () => json<{ accepted: boolean }>("due", { method: "POST" }),
};

export function githubSyncError(error: unknown): string {
  return error instanceof Error ? error.message : "The GitHub sync request failed.";
}
