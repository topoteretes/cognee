// Browser-side SDK calls: OAuth authorization must set its nonce cookie in
// this browser, not in a Next server action. localFetch includes credentials.
import localFetch from "@/modules/instances/localFetch";

export type SdkProvider = "google_drive" | "gmail" | "github";

export interface SdkConnection {
  connected: boolean;
  dataset_id?: string | null;
  stored_items?: number;
  account_label?: string | null;
  sync_status?: string | null;
  last_synced_at?: string | null;
  sync_counts?: Record<string, number> | null;
  sync_activity_unknown?: boolean;
  /** GitHub only: the run progress could not be read, so activity is not known. */
  progress_unavailable?: boolean;
  sync_progress?: SdkSyncRun[];
}

// OutDTO serializes Python field names as camelCase on the wire.
interface SdkConnectionResponse {
  connected: boolean;
  datasetId?: string | null;
  storedItems?: number | null;
  accountLabel?: string | null;
  syncStatus?: string | null;
  lastSyncedAt?: string | null;
  syncCounts?: Record<string, number> | null;
}

export interface SdkRepositoryProgress {
  source: string;
  branch: string | null;
  state: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  phase: string;
  completed: number;
  total: number;
}

export interface SdkSyncRun {
  resources?: Record<string, SdkRepositoryProgress>;
  runId: string;
  state: "running" | "waiting" | "succeeded" | "partial" | "failed" | "cancelled" | "blocked";
  phase: string;
  completed: number | null;
  total: number | null;
  unit: string | null;
  endedAt: string | null;
  counts: Record<string, number>;
}

interface SyncProgressResult {
  latest: SdkSyncRun | null;
  activeRuns: SdkSyncRun[];
  unknownActivityCount: number;
}

export interface SdkResource {
  id: string;
  name: string;
  description?: string | null;
  selected: boolean;
  attributes?: { branch?: string | null };
  /** Selected earlier but no longer returned by the provider (set by the card). */
  unavailable?: boolean;
}

export interface SdkResources {
  resources: SdkResource[];
  selected: string[] | null;
}

async function request<T>(provider: SdkProvider, path: string, options?: RequestInit): Promise<T> {
  const response = await localFetch(`/v1/integrations/${provider}/${path}`, options);
  return response.json();
}

export const sdkApi = {
  connection: async (provider: SdkProvider): Promise<SdkConnection> => {
    const response = await request<SdkConnectionResponse>(provider, "connection");
    const connection: SdkConnection = {
      connected: response.connected,
      account_label: response.accountLabel,
      dataset_id: response.datasetId,
      stored_items: response.storedItems ?? undefined,
      sync_status: response.syncStatus,
      last_synced_at: response.lastSyncedAt,
      sync_counts: response.syncCounts,
    };
    // GitHub writes durable run progress, not the Google credential sync fields.
    if (provider === "github" && response.connected) {
      // A failed progress read must not make a connected account look disconnected,
      // but it cannot be taken for "nothing is running" either: the card keeps what
      // /connection reported, says progress is unavailable and holds Refresh until a
      // later poll reads it.
      const progress = await request<SyncProgressResult>(provider, "sync/progress").catch(() => null);
      connection.progress_unavailable = progress === null;
      if (progress) {
        const latest = progress.latest;
        connection.sync_activity_unknown = progress.unknownActivityCount > 0;
        connection.sync_status = progress.activeRuns.length > 0 ? "syncing"
          : connection.sync_activity_unknown ? "unknown"
          : latest?.state === "succeeded" ? "ok"
          : latest ? "degraded" : null;
        connection.last_synced_at = latest?.endedAt;
        connection.sync_counts = latest?.counts;
        connection.sync_progress = progress.activeRuns.length > 0
          ? progress.activeRuns : latest ? [latest] : [];
      }
    }
    return connection;
  },
  authorize: async (provider: SdkProvider) => {
    const response = await request<{ authorizeUrl: string }>(provider, "authorize", { method: "POST" });
    return { authorize_url: response.authorizeUrl };
  },
  branches: (repository: string) => request<{ branches: string[] }>("github", `repositories/branches?repository=${encodeURIComponent(repository)}`),
  selectBranch: (repository: string, branch: string | null) => request<{ branch: string | null }>("github", "repositories/branches", {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ repository, branch }),
  }),
  resources: (provider: SdkProvider) => request<SdkResources>(provider, "resources"),
  select: (provider: SdkProvider, resourceIds: string[] | null) =>
    request<{ selected: string[] | null }>(provider, "resources", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resource_ids: resourceIds }),
    }),
  sync: (provider: SdkProvider) =>
    request<{ accepted: boolean }>(provider, "sync", { method: "POST" }),
  disconnect: (provider: SdkProvider, deleteData: boolean) =>
    request<{ disconnected: boolean }>(provider, `connection?delete_data=${deleteData}`, {
      method: "DELETE",
    }),
};

export function sdkError(error: unknown): string {
  return error instanceof Error ? error.message : "The integration request failed.";
}

export function sdkAuthorizeUrl(provider: SdkProvider, raw: string): string {
  const url = new URL(raw);
  const origin = provider === "github" ? "https://github.com" : "https://accounts.google.com";
  if (url.origin !== origin) {
    throw new Error("The server returned an unexpected authorization URL.");
  }
  return url.href;
}
