import fetchMock from "jest-fetch-mock";
import { sdkApi, sdkAuthorizeUrl } from "../sdkApi";

beforeEach(() => fetchMock.resetMocks());

test("authorization is a credentialed browser POST to the SDK", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ authorizeUrl: "https://accounts.google.com/o/oauth2/v2/auth" }));
  await expect(sdkApi.authorize("google_drive")).resolves.toEqual({ authorize_url: "https://accounts.google.com/o/oauth2/v2/auth" });
  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/integrations/google_drive/authorize", expect.objectContaining({ method: "POST", credentials: "include" }));
});

test.each(["google_drive", "gmail"] as const)("maps %s camelCase connection fields without changing counter keys", async (provider) => {
  fetchMock.mockResponseOnce(JSON.stringify({
    connected: true,
    accountLabel: "tester@example.com",
    syncStatus: "degraded",
    lastSyncedAt: "2026-09-22T10:00:00Z",
    syncCounts: { failed_ingestion: 1, skipped: 2 },
  }));
  await expect(sdkApi.connection(provider)).resolves.toEqual({
    connected: true,
    account_label: "tester@example.com",
    sync_status: "degraded",
    last_synced_at: "2026-09-22T10:00:00Z",
    sync_counts: { failed_ingestion: 1, skipped: 2 },
  });
});

test.each([{ selection: [] }, { selection: ["folder-a"] }, { selection: null }])("selection preserves the SDK resource_ids payload: %j", async ({ selection }) => {
  fetchMock.mockResponseOnce(JSON.stringify({ selected: selection }));
  await sdkApi.select("gmail", selection);
  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/integrations/gmail/resources", expect.objectContaining({ method: "PUT", body: JSON.stringify({ resource_ids: selection }), credentials: "include" }));
});

test.each([false, true])("deletion requires the explicit delete_data flag: %s", async (deleteData) => {
  fetchMock.mockResponseOnce(JSON.stringify({ disconnected: true }));
  await sdkApi.disconnect("gmail", deleteData);
  expect(fetchMock).toHaveBeenCalledWith(`http://localhost:8000/api/v1/integrations/gmail/connection?delete_data=${deleteData}`, expect.objectContaining({ method: "DELETE", credentials: "include" }));
});

test("server configuration failures reach the UI", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ detail: "gmail integration is not configured on this server." }), { status: 503 });
  await expect(sdkApi.authorize("gmail")).rejects.toThrow("gmail integration is not configured");
});

test("sync acceptance is not converted into completion", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ accepted: true }));
  await expect(sdkApi.sync("google_drive")).resolves.toEqual({ accepted: true });
  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/integrations/google_drive/sync", expect.objectContaining({ method: "POST" }));
});

test.each(["javascript:alert(1)", "https://accounts.google.com.evil.test/auth", "http://accounts.google.com/auth"])("refuses an unsafe popup URL: %s", (url) => expect(() => sdkAuthorizeUrl("google_drive", url)).toThrow());

test("accepts the official HTTPS Google authorization origin", () => {
  expect(sdkAuthorizeUrl("google_drive", "https://accounts.google.com/o/oauth2/v2/auth?state=test")).toContain("state=test");
});


test("GitHub authorization uses a credentialed SDK request", async () => {
  const authorizeUrl = "https://github.com/apps/cognee/installations/new?state=test";
  fetchMock.mockResponseOnce(JSON.stringify({ authorizeUrl }));
  await expect(sdkApi.authorize("github")).resolves.toEqual({ authorize_url: authorizeUrl });
  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/integrations/github/authorize", expect.objectContaining({ method: "POST", credentials: "include" }));
  expect(sdkAuthorizeUrl("github", authorizeUrl)).toBe(authorizeUrl);
});

test.each(["javascript:alert(1)", "https://github.com.evil.test/auth", "http://github.com/apps/cognee", "https://accounts.google.com/auth"])("GitHub rejects an unexpected authorization origin: %s", (url) => {
  expect(() => sdkAuthorizeUrl("github", url)).toThrow();
});


test.each([
  { state: "succeeded", active: false, unknown: 0, status: "ok" },
  { state: "failed", active: false, unknown: 0, status: "degraded" },
  { state: "partial", active: false, unknown: 0, status: "degraded" },
  { state: "running", active: true, unknown: 0, status: "syncing" },
  { state: "running", active: false, unknown: 1, status: "unknown" },
])("GitHub displays durable $state progress instead of empty credential timestamps", async ({ state, active, unknown, status }) => {
  fetchMock.mockResponseOnce(JSON.stringify({ connected: true, datasetId: "repo-dataset", storedItems: 2, accountLabel: "my-org" }));
  fetchMock.mockResponseOnce(JSON.stringify({
    latest: { state, endedAt: active || unknown ? null : "2026-10-23T15:02:24Z", counts: { processed: 2, failed: 0 } },
    activeRuns: active ? [{}] : [],
    unknownActivityCount: unknown,
  }));
  const connection = await sdkApi.connection("github");
  expect(connection).toMatchObject({ dataset_id: "repo-dataset", stored_items: 2, sync_status: status, sync_activity_unknown: unknown > 0, sync_counts: { processed: 2, failed: 0 } });
  expect(connection.last_synced_at).toBe(active || unknown ? null : "2026-10-23T15:02:24Z");
  expect(fetchMock).toHaveBeenLastCalledWith("http://localhost:8000/api/v1/integrations/github/sync/progress", expect.objectContaining({ credentials: "include" }));
});

test("disconnected GitHub does not request progress for a nonexistent connection", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ connected: false }));
  await expect(sdkApi.connection("github")).resolves.toMatchObject({ connected: false });
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

test("a connected GitHub account without any runs is not marked synced", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ connected: true }));
  fetchMock.mockResponseOnce(JSON.stringify({ latest: null, activeRuns: [], unknownActivityCount: 0 }));
  await expect(sdkApi.connection("github")).resolves.toMatchObject({ sync_status: null });
});

test("active run progress takes precedence over a newer completed invocation", async () => {
  const activeRun = { runId: "active", state: "running", phase: "processing", completed: 1, total: 3, unit: "repositories", endedAt: null, counts: {} };
  const latest = { ...activeRun, runId: "latest", state: "succeeded", completed: 3 };
  fetchMock.mockResponseOnce(JSON.stringify({ connected: true }));
  fetchMock.mockResponseOnce(JSON.stringify({ latest, activeRuns: [activeRun], unknownActivityCount: 0 }));
  await expect(sdkApi.connection("github")).resolves.toMatchObject({ sync_status: "syncing", sync_progress: [activeRun] });
});

test("completed run counters remain available for the final progress bar", async () => {
  const latest = { runId: "latest", state: "succeeded", phase: "finalizing", completed: 3, total: 3, unit: "repositories", endedAt: "2026-10-23T15:02:24Z", counts: { processed: 3 } };
  fetchMock.mockResponseOnce(JSON.stringify({ connected: true }));
  fetchMock.mockResponseOnce(JSON.stringify({ latest, activeRuns: [], unknownActivityCount: 0 }));
  await expect(sdkApi.connection("github")).resolves.toMatchObject({ sync_status: "ok", sync_progress: [latest] });
});

test("a failed GitHub progress read keeps the connection and says progress is unavailable", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ connected: true, datasetId: "repo-dataset", accountLabel: "my-org", syncStatus: "ok", lastSyncedAt: "2026-10-22T10:00:00Z" }));
  fetchMock.mockResponseOnce(JSON.stringify({ detail: "database busy" }), { status: 500 });
  const connection = await sdkApi.connection("github");
  expect(connection).toMatchObject({
    connected: true, account_label: "my-org", sync_status: "ok", last_synced_at: "2026-10-22T10:00:00Z",
    progress_unavailable: true,
  });
  expect(connection.sync_activity_unknown).toBeUndefined();
});
