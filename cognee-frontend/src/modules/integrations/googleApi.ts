// Browser-side SDK calls: OAuth authorization must set its nonce cookie in
// this browser, not in a Next server action. localFetch includes credentials.
import localFetch from "@/modules/instances/localFetch";

export type GoogleProvider = "google_drive" | "gmail";

export interface GoogleConnection {
  connected: boolean;
  dataset_id?: string | null;
  stored_items?: number;
  account_label?: string | null;
  sync_status?: string | null;
  last_synced_at?: string | null;
  sync_counts?: Record<string, number> | null;
}

// OutDTO serializes Python field names as camelCase on the wire.
interface GoogleConnectionResponse {
  connected: boolean;
  datasetId?: string | null;
  storedItems?: number | null;
  accountLabel?: string | null;
  syncStatus?: string | null;
  lastSyncedAt?: string | null;
  syncCounts?: Record<string, number> | null;
}

export interface GoogleResource {
  id: string;
  name: string;
  description?: string | null;
  selected: boolean;
}

export interface GoogleResources {
  resources: GoogleResource[];
  selected: string[] | null;
}

async function request<T>(provider: GoogleProvider, path: string, options?: RequestInit): Promise<T> {
  const response = await localFetch(`/v1/integrations/${provider}/${path}`, options);
  return response.json();
}

export const googleApi = {
  connection: async (provider: GoogleProvider): Promise<GoogleConnection> => {
    const response = await request<GoogleConnectionResponse>(provider, "connection");
    return {
      connected: response.connected,
      account_label: response.accountLabel,
      dataset_id: response.datasetId,
      stored_items: response.storedItems ?? undefined,
      sync_status: response.syncStatus,
      last_synced_at: response.lastSyncedAt,
      sync_counts: response.syncCounts,
    };
  },
  authorize: async (provider: GoogleProvider) => {
    const response = await request<{ authorizeUrl: string }>(provider, "authorize", { method: "POST" });
    return { authorize_url: response.authorizeUrl };
  },
  resources: (provider: GoogleProvider) => request<GoogleResources>(provider, "resources"),
  select: (provider: GoogleProvider, resourceIds: string[] | null) =>
    request<{ selected: string[] | null }>(provider, "resources", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resource_ids: resourceIds }),
    }),
  sync: (provider: GoogleProvider) =>
    request<{ accepted: boolean }>(provider, "sync", { method: "POST" }),
  disconnect: (provider: GoogleProvider, deleteData: boolean) =>
    request<{ disconnected: boolean }>(provider, `connection?delete_data=${deleteData}`, {
      method: "DELETE",
    }),
};

export function googleError(error: unknown): string {
  return error instanceof Error ? error.message : "The Google integration request failed.";
}

export function googleAuthorizeUrl(raw: string): string {
  const url = new URL(raw);
  if (url.origin !== "https://accounts.google.com") {
    throw new Error("The server returned an unexpected Google authorization URL.");
  }
  return url.href;
}
