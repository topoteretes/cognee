import fetchMock from "jest-fetch-mock";
import { googleApi, googleAuthorizeUrl } from "../googleApi";

jest.mock("@/modules/users/getLocalApiUrl", () => ({ getLocalApiUrl: () => "http://localhost:8000" }));
beforeEach(() => fetchMock.resetMocks());

test("authorization is a credentialed browser POST to the SDK", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ authorizeUrl: "https://accounts.google.com/o/oauth2/v2/auth" }));
  await expect(googleApi.authorize("google_drive")).resolves.toEqual({ authorize_url: "https://accounts.google.com/o/oauth2/v2/auth" });
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
  await expect(googleApi.connection(provider)).resolves.toEqual({
    connected: true,
    account_label: "tester@example.com",
    sync_status: "degraded",
    last_synced_at: "2026-09-22T10:00:00Z",
    sync_counts: { failed_ingestion: 1, skipped: 2 },
  });
});

test.each([{ selection: [] }, { selection: ["folder-a"] }, { selection: null }])("selection preserves the SDK resource_ids payload: %j", async ({ selection }) => {
  fetchMock.mockResponseOnce(JSON.stringify({ selected: selection }));
  await googleApi.select("gmail", selection);
  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/integrations/gmail/resources", expect.objectContaining({ method: "PUT", body: JSON.stringify({ resource_ids: selection }), credentials: "include" }));
});

test.each([false, true])("deletion requires the explicit delete_data flag: %s", async (deleteData) => {
  fetchMock.mockResponseOnce(JSON.stringify({ disconnected: true }));
  await googleApi.disconnect("gmail", deleteData);
  expect(fetchMock).toHaveBeenCalledWith(`http://localhost:8000/api/v1/integrations/gmail/connection?delete_data=${deleteData}`, expect.objectContaining({ method: "DELETE", credentials: "include" }));
});

test("server configuration failures reach the UI", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ detail: "gmail integration is not configured on this server." }), { status: 503 });
  await expect(googleApi.authorize("gmail")).rejects.toThrow("gmail integration is not configured");
});

test("sync acceptance is not converted into completion", async () => {
  fetchMock.mockResponseOnce(JSON.stringify({ accepted: true }));
  await expect(googleApi.sync("google_drive")).resolves.toEqual({ accepted: true });
  expect(fetchMock).toHaveBeenCalledWith("http://localhost:8000/api/v1/integrations/google_drive/sync", expect.objectContaining({ method: "POST" }));
});

test.each(["javascript:alert(1)", "https://accounts.google.com.evil.test/auth", "http://accounts.google.com/auth"])("refuses an unsafe popup URL: %s", (url) => expect(() => googleAuthorizeUrl(url)).toThrow());

test("accepts the official HTTPS Google authorization origin", () => {
  expect(googleAuthorizeUrl("https://accounts.google.com/o/oauth2/v2/auth?state=test")).toContain("state=test");
});
